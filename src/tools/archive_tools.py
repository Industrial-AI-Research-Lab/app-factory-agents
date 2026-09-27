"""Agent-facing retrieval tools for archived oversized results (AppFactory-148).

Three tools over the refs that ArchiveStore.maybe_spill leaves in a
conversation:

- ``archive_inspect`` — metadata + structural preview, answered from the Mongo
  locator record alone (zero S3 traffic);
- ``archive_query`` — regex over the normalized line-oriented twin, streamed
  from object storage with constant memory and explicit bounds;
- ``archive_fetch`` — downloads into the sandbox at
  ``/tmp/AppFactory-archive/<ref>`` for local grep/jq/python processing of objects
  too big to route through the backend (design target 20 GB on an 8 GB
  backend). It returns only once the file is on disk, so there is nothing to
  poll. The backend mints a presigned GET that lives just longer than the
  download it authorizes and hands it to the SANDBOX inside the command; the
  tool result carries only the path and byte count — the URL never enters the
  LLM context (captures, summaries and chat all persist that context, so a URL
  there is a leaked capability).

  It does still ride in on the command line, and container-use writes every
  command it runs into a git note on the host, which ``cu log`` prints back.
  Signed URLs are stripped from that output before anything publishes it
  (``sandbox/container_manager.scrub_signed_urls``), which covers the API
  response, the automatic run-end snapshot in Mongo and the UI — but NOT the
  note on the host's own disk. Keeping it off the command line altogether needs
  the URL written to a file that curl reads with ``-K``; that is not done yet
  because a file written to the wrong place is committed to the host repo
  permanently, which is worse than the note, and confirming where container-use
  puts it needs a live sandbox.

Attach policy: these schemas are injected per-request only when the (project,
run) already has at least one archive ref — deliberately outside the per-agent
allowlist for discoverability, so tool definitions stay untouched for the 99%
of runs that never spill (see generic_agent). ``archive_fetch`` is gated on the
agent's effective allowlist containing ``bash`` or ``read`` (AppFactory-315):
without a sandbox file reader the download is unusable, so tool-only agents
get inspect+query only.

Ownership: every entry point resolves the ref and requires
``record.project_id == caller's project`` BEFORE any storage work. A presigned
URL is a bearer capability — minted for the wrong object it bypasses every
later check for its whole TTL — so the order here is the security invariant,
not a style choice. ``record.tenant_id`` is never the authority (absent
tenants collapse to ``__root__``; ADR-0005).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sys
from copy import deepcopy
from typing import Any, Dict, Optional

# NOT stdlib re, deliberately: agent-supplied patterns can backtrack
# catastrophically, and a running re.search holds the GIL and never returns
# to the event loop — asyncio.wait_for cannot fire and even a worker thread
# doesn't free the loop (probed: heartbeat starved either way). The regex
# module bounds each match internally via its timeout parameter.
import regex

from storage.archive_store import DEFAULT_SPILL_THRESHOLD_BYTES

logger = logging.getLogger(__name__)

ARCHIVE_TOOL_IDS = ("archive_inspect", "archive_query", "archive_fetch")
# archive_fetch drops a file in the sandbox; only useful when the agent can
# open that file (bash and/or read). Attach policy gates on these ids in the
# agent's effective allowlist (AppFactory-315).
_SANDBOX_FILE_TOOL_IDS = frozenset({"bash", "read"})

DEFAULT_QUERY_TIMEOUT_SECONDS = 120.0
DEFAULT_QUERY_MAX_SCAN_BYTES = 256 * 1024 * 1024
# Headroom between the scan budget and the backend's deadline over the whole
# call, so the outer timer only ever fires on work the scan budget cannot see.
_QUERY_CEILING_MARGIN_SECONDS = 60
# A backstop against a wedged transfer, not a service level — a multi-GB
# archive over a slow link is a legitimate download. An hour covers the 20 GB
# design target at ~6 MB/s; a day covered it at 240 KB/s and cost a day of a
# held sandbox when an endpoint stalled instead. Deployments do not set the
# env var (checked), so this number is the one that actually applies, and two
# things scale with it: how long a stalled fetch parks the run, and how long
# the signed URL it minted stays usable.
DEFAULT_FETCH_TIMEOUT_SECONDS = 3_600.0
# How much longer than the download its signed URL stays valid. Derived from the
# budget rather than set on its own: the URL exists only to authorize this one
# transfer, so a separate setting could only ever contradict it — and whichever
# was shorter would silently win.
_FETCH_TTL_SLACK_SECONDS = 60
# The backend's own deadline, sitting above curl's. Twice the budget because the
# retry window plus one last attempt can reach that; it must never fire first,
# which would kill downloads that were going to succeed.
_FETCH_CEILING_FACTOR = 2
_FETCH_CEILING_MARGIN_SECONDS = 300

_PATTERN_MAX_CHARS = 512
_DEFAULT_MAX_MATCHES = 50
_MAX_MATCHES_CAP = 200
# Hard per-line matching budget. One slow line stalls the loop for at most
# this long; the caller's overall wait_for handles totals. Lines that hit it
# are counted and reported as possibly-missed, not silently skipped.
_LINE_MATCH_TIMEOUT_SECONDS = 0.1
# regex EXPANDS bounded repeats at COMPILE time, and that expansion holds the
# GIL with no timeout hook (regex.compile rejects a timeout kwarg), so no
# asyncio deadline and no thread offload can interrupt it — and a deep enough
# expansion overflows the C stack, which kills the process outright rather than
# raising. The per-line match timeout guards MATCHING, not this.
#
# Counting repeats CANNOT decide this: the cost is the counts times the size of
# what is repeated, and measuring one without the other says nothing. Probed:
# `(x{300}){300}` and `((90-way alternation){300}){300}` have the identical
# product of 90_000, and compile in 0.03s and never respectively — the second
# dies with a stack overflow. Sizing the body instead fails the other way (a
# 488-char body repeated 100_000 times compiles in 0.3s), because the engine
# does not expand every shape. So the pattern is not modelled at all: anything
# that CAN expand is compiled in a throwaway process first, where a wall clock
# and a memory cap apply and a stack overflow costs us a child we were ready to
# lose. The product below stays only as a free pre-filter that rejects the
# absurd without paying for a process.
_MAX_REPEAT_PRODUCT = 100_000
# Vetting budget for a pattern that can expand. Generous next to a real compile
# (0.005s is the worst brace-free case measured) and cheap next to the scan it
# admits; a pattern needing longer than this is one we do not want to run.
_COMPILE_PROBE_SECONDS = 2.0
_COMPILE_PROBE_MEMORY_BYTES = 1024 * 1024 * 1024
# Runs in a child: cap the address space so a greedy expansion raises
# MemoryError there instead of pushing the pod toward the OOM killer, then
# compile exactly what the parent would.
_COMPILE_PROBE_SOURCE = """
import sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (%d, %d))
except Exception:
    pass
import regex
regex.compile(sys.stdin.buffer.read().decode("utf-8", "replace"))
""" % (_COMPILE_PROBE_MEMORY_BYTES, _COMPILE_PROBE_MEMORY_BYTES)
_BOUNDED_REPEAT_RE = regex.compile(r"\{(\d+)(?:,(\d*))?\}")
# Public: tool_dispatcher exempts this path from its bash-grep redirect (the
# indexed grep tool's walk is rooted in the workdir and can never reach these
# files, so "use the grep tool" would be a dead end here).
ARCHIVE_FETCH_DIR = "/tmp/AppFactory-archive"
_STDERR_SNIPPET_MAX = 300

# One error for both "no such ref" and "someone else's ref": a distinguishable
# answer would let guessed ref_ids enumerate other projects' archives.
_DENIED_ERROR = "unknown or inaccessible ref_id"

_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "archive_inspect",
            "description": (
                "Inspect an archived oversized tool result by its ref_id: sizes, "
                "content type, sha256, head/tail preview and available "
                "representations. Cheap metadata-only call — use it first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ref_id": {"type": "string", "description": "The archive ref id (arch_...)."},
                },
                "required": ["ref_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archive_query",
            "description": (
                "Regex-search an archived result line by line without loading it "
                "into the conversation. Returns matching lines with a total count. "
                "Works on the normalized representation (text / NDJSON / pretty "
                "JSON). For heavy processing of the whole object use archive_fetch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ref_id": {"type": "string", "description": "The archive ref id (arch_...)."},
                    "pattern": {"type": "string", "description": "Regular expression matched against each line."},
                    "max_matches": {
                        "type": "integer",
                        "description": f"Max matching lines to return (default {_DEFAULT_MAX_MATCHES}, cap {_MAX_MATCHES_CAP}).",
                    },
                },
                "required": ["ref_id", "pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archive_fetch",
            "description": (
                "Download an archived result into your sandbox as a file, and return "
                f"its path under {ARCHIVE_FETCH_DIR}. The download is finished when the "
                "call returns — there is nothing to poll. Requires shell tooling to be "
                "useful: you get a file, and you choose how to process it. If you only "
                "need to search the content, use archive_query instead — it runs "
                "server-side and needs no download. Never read the whole file into the "
                "conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ref_id": {"type": "string", "description": "The archive ref id (arch_...)."},
                    "representation": {
                        "type": "string",
                        "enum": ["raw", "normalized"],
                        "description": "Which stored representation to download (default raw — the byte-exact original).",
                    },
                },
                "required": ["ref_id"],
            },
        },
    },
]


_QUERY_DESCRIPTION_NO_FETCH = (
    "Regex-search an archived result line by line without loading it "
    "into the conversation. Returns matching lines with a total count. "
    "Works on the normalized representation (text / NDJSON / pretty "
    "JSON). Prefer targeted patterns over broad scans of large objects."
)


def agent_can_archive_fetch(effective_allowed_tool_ids: Any) -> bool:
    """True when the agent can open a sandbox file (bash and/or read)."""
    if not effective_allowed_tool_ids:
        return False
    present = {str(tid).strip() for tid in effective_allowed_tool_ids if tid}
    return not _SANDBOX_FILE_TOOL_IDS.isdisjoint(present)


def archive_tool_schemas(*, include_fetch: bool = True) -> list[Dict[str, Any]]:
    """OpenAI function-calling schemas for the archive tools (fresh copies —
    callers splice them into per-request tool lists).

    ``include_fetch=False`` omits ``archive_fetch`` and rewrites ``archive_query``
    so the model is not told to download (AppFactory-315 tool-only agents).
    """
    schemas = deepcopy(_SCHEMAS)
    if include_fetch:
        return schemas
    out: list[Dict[str, Any]] = []
    for schema in schemas:
        name = (schema.get("function") or {}).get("name")
        if name == "archive_fetch":
            continue
        if name == "archive_query":
            schema["function"]["description"] = _QUERY_DESCRIPTION_NO_FETCH
        out.append(schema)
    return out


def _env_float(name: str, default: float) -> float:
    # Floors non-positive values like every sibling reader (routes'
    # _download_ttl_seconds, ArchiveStore.from_env): TTL=0 mints an
    # already-expired URL, TIMEOUT=0 fails every query instantly.
    try:
        raw = os.getenv(name)
        value = float(raw) if raw else default
    except ValueError:
        logger.warning("[ARCHIVE_TOOLS] invalid %s=%r — using default %s", name, os.getenv(name), default)
        return default
    if value <= 0:
        logger.warning("[ARCHIVE_TOOLS] non-positive %s=%r — using default %s", name, os.getenv(name), default)
        return default
    return value


def _fetch_budget_seconds(override: Optional[float] = None) -> float:
    """Seconds curl may spend on one archive_fetch.

    Floored at 1 because curl reads `--max-time 0` as "no limit" — without the
    floor, the tightest setting an operator can express becomes an unbounded
    download once int() truncates it.
    """
    raw = override if override is not None else _env_float(
        "ARCHIVE_FETCH_TIMEOUT_SECONDS", DEFAULT_FETCH_TIMEOUT_SECONDS
    )
    return max(1.0, float(raw))


def _query_timeout_seconds(override: Optional[float] = None) -> float:
    """Seconds archive_query may spend scanning."""
    if override is not None:
        return float(override)
    return _env_float("ARCHIVE_QUERY_TIMEOUT_SECONDS", DEFAULT_QUERY_TIMEOUT_SECONDS)


def query_ceiling_seconds() -> float:
    """How long the backend waits for the whole archive_query call.

    The scan budget covers the scan and nothing else. The ownership lookup runs
    before it — a database round-trip with no deadline of its own — so a wedged
    Mongo parks the run on a tool that looks time-bounded from the outside. This
    sits above the scan budget so it can only fire on the parts that budget
    never sees.
    """
    return _query_timeout_seconds() + _QUERY_CEILING_MARGIN_SECONDS


def fetch_ceiling_seconds() -> float:
    """How long the backend waits for archive_fetch before giving up on it.

    curl's own limit only fires if the sandbox is alive and running the shell we
    handed it. A sandbox that is still up but has stopped answering leaves the
    backend waiting on a reply that never arrives, and the layers in between
    have no deadline of their own. This is the one timer that survives that,
    because none of it lives inside the container.
    """
    return _fetch_budget_seconds() * _FETCH_CEILING_FACTOR + _FETCH_CEILING_MARGIN_SECONDS


def _bounded_repeat_product(pattern: str) -> int:
    """Upper bound on the compile-time node blow-up of a pattern's bounded
    repeats. Multiplies the max count of every {n}/{n,}/{n,m}. This is a
    DELIBERATE over-estimate: it multiplies SIBLING repeats too (which only
    add), so it can over-reject (e.g. `.{0,512}X.{0,512}` = 262k), never
    under-reject — the safe direction for an adversarial guard. Escaped braces
    (`\\{99\\}`) are also over-counted; harmless for the same reason. Bails
    once it passes the cap so its own loop can't be made expensive."""
    product = 1
    for m in _BOUNDED_REPEAT_RE.finditer(pattern):
        low, high = m.group(1), m.group(2)
        # {n} → n; {n,m} → m; {n,} open-ended → n (the loop back-edge adds no
        # bounded expansion beyond the n unrolled copies).
        count = int(low) if high in (None, "") else int(high)
        product *= max(count, 1)  # {0} must not zero the running product
        if product > _MAX_REPEAT_PRODUCT:
            break
    return product


class _MatchFailure(Exception):
    """The regex engine failed on a line — a pattern problem, not a read problem."""


async def _compiles_safely(pattern: str) -> bool:
    """Whether the parent may compile this pattern, decided by compiling it in a
    child that we can kill.

    Only called for patterns that contain a bounded repeat, so the process cost
    lands on the shapes that can expand and never on ordinary ones. The child is
    the whole point: a compile that overflows the C stack takes its process down
    with no exception to catch, so the process it takes down has to be one whose
    death is an answer rather than an outage. Waiting on it is ordinary async, so
    the loop keeps serving other runs throughout — which is what compiling here
    would prevent.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", _COMPILE_PROBE_SOURCE,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except Exception as exc:
        # No child to vet with, and this pattern can expand. Refusing is the
        # only safe answer left; brace-free patterns never reach here, so the
        # ordinary case keeps working even on a host that cannot spawn.
        logger.warning("[ARCHIVE_TOOLS] cannot vet pattern out of process: %s", exc)
        return False

    try:
        await asyncio.wait_for(
            proc.communicate(pattern.encode("utf-8", errors="replace")),
            timeout=_COMPILE_PROBE_SECONDS,
        )
    except asyncio.TimeoutError:
        _terminate(proc)
        await proc.wait()
        return False
    except Exception:
        _terminate(proc)
        await proc.wait()
        return False
    # Non-zero covers every way the child can lose: regex.error, MemoryError
    # from the address-space cap, and the signal or fatal status left by a
    # stack overflow.
    return proc.returncode == 0


def _terminate(proc: "asyncio.subprocess.Process") -> None:
    """Best-effort kill of a probe child that already exited on its own."""
    try:
        proc.kill()
    except ProcessLookupError:
        pass


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name)
        value = int(raw) if raw else default
    except ValueError:
        logger.warning("[ARCHIVE_TOOLS] invalid %s=%r — using default %s", name, os.getenv(name), default)
        return default
    if value <= 0:
        logger.warning("[ARCHIVE_TOOLS] non-positive %s=%r — using default %s", name, os.getenv(name), default)
        return default
    return value


def _trailing_byte_count(stdout: str) -> Optional[int]:
    """The `wc -c` count from sandbox output, or None if it never printed one.

    container-use appends its own prose trailer ("Any changes to the container
    workdir…") after the command's output, so the last line is not the count;
    scan upward for the last all-digit line instead. The isascii() guard keeps
    int() from raising on characters like '²', which isdigit() accepts.
    """
    for line in reversed(stdout.splitlines()):
        token = line.strip()
        if token.isascii() and token.isdigit():
            return int(token)
    return None


def _fetch_soft_error(message: str, exec_res: Any = None) -> Dict[str, Any]:
    """Sandbox download soft-fail with infra labels for the terminal gate.

    execute_in_container may already set error_type (e.g. unavailable); keep it.
    Untyped soft-fail must not look like domain tool failure (COMPLETED after done).
    """
    out: Dict[str, Any] = {"status": "error", "error": message}
    if isinstance(exec_res, dict):
        if exec_res.get("error_type"):
            out["error_type"] = exec_res["error_type"]
        if exec_res.get("outcome_unknown"):
            out["outcome_unknown"] = exec_res["outcome_unknown"]
        if "exit_code" in exec_res:
            out["exit_code"] = exec_res["exit_code"]
    if "error_type" not in out and "outcome_unknown" not in out:
        out["error_type"] = "unavailable"
    return out


async def _authorized_record(archive_store, project_id: Optional[str], ref_id: Any) -> Optional[Dict[str, Any]]:
    """The record, or None for missing/foreign/blank — indistinguishably."""
    if not project_id or not isinstance(ref_id, str) or not ref_id.strip():
        return None
    record = await archive_store.get_ref(ref_id.strip())
    if not record or record.get("project_id") != project_id:
        return None
    return record


def _denied() -> Dict[str, Any]:
    return {"status": "error", "error": _DENIED_ERROR}


def _is_missing_key_error(exc: Exception) -> bool:
    resp = getattr(exc, "response", None)
    if not isinstance(resp, dict):
        return False
    code = str(((resp.get("Error") or {}).get("Code")) or "")
    return code in ("404", "NoSuchKey", "NotFound")


# ---------------------------------------------------------------------------
# archive_inspect
# ---------------------------------------------------------------------------


async def inspect_ref(archive_store, project_id: Optional[str], params: Dict[str, Any]) -> Dict[str, Any]:
    record = await _authorized_record(archive_store, project_id, (params or {}).get("ref_id"))
    if record is None:
        return _denied()

    # Representations are surfaced without their object keys — locators stay
    # server-side; the agent addresses everything by ref_id.
    representations: Dict[str, Any] = {}
    for name, rep in (record.get("representations") or {}).items():
        if not isinstance(rep, dict):
            continue
        representations[name] = {
            k: rep[k]
            for k in ("content_type", "size_bytes", "kind", "source_field", "omitted_fields")
            if rep.get(k) is not None
        }

    out: Dict[str, Any] = {
        "status": "success",
        "ref_id": record["_id"],
        "representations": representations,
    }
    for field in ("size_bytes", "content_type", "sha256", "est_tokens", "preview", "tool_id", "agent_id", "run_id"):
        if record.get(field) is not None:
            out[field] = record[field]
    created = record.get("created_at")
    if created is not None:
        out["created_at"] = created.isoformat() if hasattr(created, "isoformat") else str(created)

    if "normalized" in representations:
        out["note"] = (
            "use archive_query(ref_id, pattern) to search this result line by line, "
            "or archive_fetch(ref_id) to download it into the sandbox"
        )
    else:
        # Pre-148 spill or a failed normalized PUT: line search has nothing to
        # scan — the raw single-line blob is only usable via a sandbox download.
        out["note"] = (
            "no normalized representation exists for this ref; use "
            "archive_fetch(ref_id) and process the raw file in the sandbox"
        )
    return out


# ---------------------------------------------------------------------------
# archive_query
# ---------------------------------------------------------------------------


async def query_ref(
    archive_store,
    project_id: Optional[str],
    params: Dict[str, Any],
    *,
    timeout_seconds: Optional[float] = None,
    max_scan_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    params = params or {}
    record = await _authorized_record(archive_store, project_id, params.get("ref_id"))
    if record is None:
        return _denied()

    pattern = params.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        return {"status": "error", "error": "archive_query: missing required param 'pattern'"}
    if len(pattern) > _PATTERN_MAX_CHARS:
        return {"status": "error", "error": f"archive_query: pattern longer than {_PATTERN_MAX_CHARS} chars"}
    repeat_product = _bounded_repeat_product(pattern)
    # A pattern with no bounded repeat cannot expand at compile time, so it goes
    # straight to the compiler — that is nearly every real pattern, and it pays
    # nothing for this guard. Measured: the worst brace-free shapes (250-deep
    # nesting, 100-way alternation, stacked quantifiers) compile in 0.005s.
    if repeat_product > 1:
        if repeat_product > _MAX_REPEAT_PRODUCT or not await _compiles_safely(pattern):
            return {
                "status": "error",
                "error": (
                    "archive_query: pattern is too expensive to compile; "
                    "simplify it (use unbounded quantifiers like .* or + instead of "
                    "large {n} counts — matching is time-bounded, so they are safe)"
                ),
            }
    try:
        rx = regex.compile(pattern)
    # NOT just regex.error: a compile that slips the product guard can still
    # raise MemoryError/RecursionError, and str(MemoryError()) is '' — a bare
    # re-raise would reach the executor catch-all and return an empty error.
    except Exception as exc:
        detail = str(exc) or type(exc).__name__
        return {"status": "error", "error": f"archive_query: invalid or too-complex regex: {detail}"}

    match_timeouts = 0

    def _bounded_match(text: str) -> bool:
        nonlocal match_timeouts
        try:
            return rx.search(text, timeout=_LINE_MATCH_TIMEOUT_SECONDS) is not None
        except TimeoutError:
            # Treated as non-matching but COUNTED — the answer reports these
            # lines as possibly missed instead of folding them into a clean 0.
            match_timeouts += 1
            return False
        except Exception as exc:
            # Anything else the engine raises mid-match is about the pattern, not
            # the object. Without this it reached the catch-all below and came
            # back as "storage read failed", sending the agent to retry a read
            # that was working fine.
            raise _MatchFailure(str(exc) or type(exc).__name__) from exc

    raw_mm = params.get("max_matches")
    try:
        # `or` would swallow an explicit 0 into the default; None-check keeps
        # 0 clamping to 1 instead of silently becoming 50.
        max_matches = int(raw_mm) if raw_mm is not None else _DEFAULT_MAX_MATCHES
    except (TypeError, ValueError):
        max_matches = _DEFAULT_MAX_MATCHES
    max_matches = max(1, min(max_matches, _MAX_MATCHES_CAP))

    rep = (record.get("representations") or {}).get("normalized")
    if not isinstance(rep, dict) or not rep.get("object_key"):
        return {
            "status": "error",
            "error": (
                "archive_query: this ref has no normalized line-oriented "
                "representation (archived before normalization existed, or its "
                "write failed); use archive_fetch to download and process the "
                "raw object in the sandbox"
            ),
        }

    timeout = _query_timeout_seconds(timeout_seconds)
    scan_cap = max_scan_bytes if max_scan_bytes is not None else _env_int(
        "ARCHIVE_QUERY_MAX_SCAN_BYTES", DEFAULT_QUERY_MAX_SCAN_BYTES
    )

    try:
        scan = await asyncio.wait_for(
            archive_store.scan_lines(
                rep["object_key"],
                _bounded_match,
                max_matches=max_matches,
                max_scan_bytes=scan_cap,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return {
            "status": "error",
            "error": (
                f"archive_query: timed out after {timeout:.0f}s; narrow the "
                "pattern, or use archive_fetch and search in the sandbox"
            ),
        }
    except _MatchFailure as exc:
        return {
            "status": "error",
            "error": f"archive_query: pattern failed while matching: {exc}",
        }
    except Exception as exc:
        if _is_missing_key_error(exc):
            return {
                "status": "error",
                "error": (
                    "archive_query: the archived object is no longer in storage "
                    "(most likely expired by retention); its metadata remains via archive_inspect"
                ),
            }
        logger.error("[ARCHIVE_TOOLS] query scan failed for %s: %s", record["_id"], exc)
        return {"status": "error", "error": "archive_query: storage read failed"}

    # The answer must fit inline: past the spill threshold the dispatcher would
    # archive the query result itself and the agent would get a ref instead of
    # its matches (200 matches × 2000 multi-byte chars ≈ 800 KB). max_matches
    # bounds count, not bytes, so trim by serialized size; at least one match
    # always survives (a single match is ≤ ~8 KB by the text cap).
    byte_budget = max(1, int(getattr(archive_store, "threshold_bytes", DEFAULT_SPILL_THRESHOLD_BYTES)) // 2)
    shown = []
    used = 0
    for m in scan["matches"]:
        sz = len(json.dumps(m, ensure_ascii=False, default=str).encode("utf-8", errors="replace"))
        if shown and used + sz > byte_budget:
            break
        shown.append(m)
        used += sz

    out: Dict[str, Any] = {
        "status": "success",
        "ref_id": record["_id"],
        "pattern": pattern,
        "matches": shown,
        "total_matches": scan["total_matches"],
        "lines_scanned": scan["lines_scanned"],
        "scanned_bytes": scan["scanned_bytes"],
        "scan_complete": scan["scan_complete"],
        "representation_kind": rep.get("kind"),
    }
    if rep.get("source_field") is not None:
        out["source_field"] = rep["source_field"]

    notes = []
    if len(shown) < len(scan["matches"]):
        notes.append(
            f"showing first {len(shown)} of {scan['total_matches']} matches "
            "(response byte cap); narrow the pattern or use archive_fetch"
        )
    elif scan["total_matches"] > len(shown):
        notes.append(f"showing first {len(shown)} of {scan['total_matches']} matches")
    # A text twin is one field verbatim — say what was searched and what
    # wasn't, or a zero here reads as "not present in the result at all".
    omitted = rep.get("omitted_fields")
    if omitted:
        notes.append(
            f"searched only the '{rep.get('source_field')}' field of the original result; "
            f"not in the searchable copy: {', '.join(str(f) for f in omitted)} — "
            "use archive_fetch(representation='raw') for the full result"
        )
    truncated_lines = scan.get("lines_truncated")
    if truncated_lines:
        out["lines_truncated"] = truncated_lines
        notes.append(
            f"{truncated_lines} line(s) exceeded the per-line scan cap and were matched "
            "only against their first slice — a match past that point would be "
            "missed; use archive_fetch for an exhaustive pass"
        )
    if match_timeouts:
        out["lines_match_timeout"] = match_timeouts
        notes.append(
            f"{match_timeouts} line(s) hit the per-line match budget and were treated "
            "as non-matching — a match there may have been missed; simplify the "
            "pattern (avoid nested quantifiers) or use archive_fetch"
        )
    if rep.get("kind") in ("pretty_json", "ndjson"):
        notes.append(
            "the searchable copy is JSON-encoded: newlines inside string values "
            "are \\n escape sequences on a single line, so ^/$ anchors match "
            "encoded lines, not original content lines"
        )
    if not scan["scan_complete"]:
        notes.append(
            "scan stopped at the byte budget before the end of the object — "
            "counts are partial; use archive_fetch for an exhaustive pass"
        )
    if notes:
        out["note"] = "; ".join(notes)
    return out


# ---------------------------------------------------------------------------
# archive_fetch
# ---------------------------------------------------------------------------


async def fetch_ref(
    archive_store,
    container_manager,
    project_id: Optional[str],
    params: Dict[str, Any],
    *,
    budget_seconds: Optional[float] = None,
    storage=None,
) -> Dict[str, Any]:
    params = params or {}
    record = await _authorized_record(archive_store, project_id, params.get("ref_id"))
    if record is None:
        return _denied()

    representation = params.get("representation") or "raw"
    if representation not in ("raw", "normalized"):
        return {"status": "error", "error": "archive_fetch: representation must be 'raw' or 'normalized'"}

    rep = (record.get("representations") or {}).get(representation)
    if isinstance(rep, dict) and rep.get("object_key"):
        object_key = rep["object_key"]
        size_bytes = rep.get("size_bytes")
    elif representation == "raw" and record.get("object_key"):
        # 183-era record: no representations map, but the raw locator exists.
        object_key = record["object_key"]
        size_bytes = record.get("size_bytes")
    else:
        return {
            "status": "error",
            "error": f"archive_fetch: no '{representation}' representation stored for this ref",
        }

    if container_manager is None:
        return {
            "status": "error",
            "error": "archive_fetch: no sandbox in this deployment; use archive_query instead",
            "error_type": "unavailable",
        }
    status = await container_manager.get_container_status(project_id)
    if not (status.get("active") and status.get("environment_id")):
        # Start one rather than refuse. The results big enough to archive
        # typically come from external MCP servers, and such a run can reach this
        # tool having never touched the sandbox — refusing there would make
        # archive_fetch unreachable in the very case it exists for. The executor
        # skips its own provisioning for this tool, so it happens here, inside
        # the deadline that already covers the download.
        try:
            created = await container_manager.get_or_create_container(project_id)
        except Exception as exc:
            logger.error(
                "[ARCHIVE_TOOLS] fetch could not start a sandbox for %s: %s", project_id, exc
            )
            created = None
        if not (created or {}).get("environment_id"):
            return {
                "status": "error",
                "error": (
                    "archive_fetch: could not start a sandbox for this project — "
                    "there is nowhere to download to; use archive_query / archive_inspect instead"
                ),
                "error_type": "unavailable",
            }
        # Put the project's files back, exactly as the executor does when IT
        # creates a container. This tool does not need them — it writes outside
        # the workdir — but creating the container is what makes every later
        # tool see one as ready, and those tools do need them. Skipping this
        # would hand the next step an empty workdir with nothing left to notice
        # it was never filled.
        if storage is not None:
            try:
                restored = await container_manager.restore_container_from_context(
                    project_id, storage
                )
                logger.info(
                    "[ARCHIVE_TOOLS] fetch started a sandbox for %s, restored %s files",
                    project_id, restored,
                )
            except Exception as exc:
                logger.warning(
                    "[ARCHIVE_TOOLS] fetch could not restore files for %s: %s", project_id, exc
                )

    # Existence BEFORE minting: presigning a vanished object would hand the
    # agent a dead capability and a confusing curl 404 inside the sandbox.
    # Both S3 calls are wrapped: botocore error text embeds the full request
    # URL (endpoint host, bucket, tenant/project/run/ref key), and an
    # unwrapped exception reaches the executor catch-all, which str()s it
    # into model context — same invariant as maybe_spill's put handler.
    try:
        head = await archive_store.head_blob(object_key)
    except Exception as exc:
        logger.error("[ARCHIVE_TOOLS] fetch head failed for %s: %s", record["_id"], exc)
        return {"status": "error", "error": "archive_fetch: archive storage unavailable", "error_type": "unavailable"}
    if head is None:
        return {
            "status": "error",
            "error": (
                "archive_fetch: the archived object is no longer in storage "
                "(most likely expired by retention); its metadata remains via archive_inspect"
            ),
        }

    budget = _fetch_budget_seconds(budget_seconds)
    # Sized to the transfer it authorizes. A TTL configured apart from the
    # budget could be shorter than the download, and the shorter number wins
    # silently — a 24h budget under a stale 10-minute TTL is really 10 minutes.
    # The same factor the fetch ceiling uses, for the same reason: --retry-max-time
    # only bounds when the LAST attempt may START, so that attempt can begin just
    # under the budget and run a full budget more. Signing for one budget would
    # let the signature die mid-transfer on exactly the slow, retried downloads
    # this budget exists to allow.
    ttl = int(budget) * _FETCH_CEILING_FACTOR + _FETCH_TTL_SLACK_SECONDS
    try:
        url = await archive_store.presign_get(object_key, ttl)
    except Exception as exc:
        logger.error("[ARCHIVE_TOOLS] fetch presign failed for %s: %s", record["_id"], exc)
        return {"status": "error", "error": "archive_fetch: failed to prepare the download", "error_type": "unavailable"}

    dest = f"{ARCHIVE_FETCH_DIR}/{record['_id']}" + (".norm" if representation == "normalized" else "")
    q = shlex.quote
    # Graded on the byte count, not the exit code: the sandbox rewrites the
    # shell's status, so a curl that wrote nothing still comes back 0 (probed).
    # `;` before wc rather than `&&` — the count must run even when curl fails,
    # because a missing file printing 0 IS the failure signal.
    # Downloading outside /workdir is deliberate: a file written there is
    # committed to a git repo on the host and survives `rm` forever (measured).
    # Both curl limits are required. --max-time bounds ONE attempt and --retry
    # restarts that clock, so --retry alone permits four full budgets: measured,
    # a 2s budget against a stalled server ran 15s over 4 connections.
    # --retry-max-time bounds the retry window itself.
    command = (
        f"mkdir -p {q(ARCHIVE_FETCH_DIR)} && rm -f {q(dest)} && "
        f"curl -fsSL --retry 3 --retry-max-time {int(budget)} "
        f"--max-time {int(budget)} -o {q(dest)} {q(url)}; "
        f"wc -c < {q(dest)} 2>/dev/null || echo 0"
    )
    exec_res = await container_manager.execute_in_container(project_id, command)
    # Scrub BEFORE truncating: the presigned URL is longer than the snippet cap,
    # so cutting first can sever it — then `.replace(url, …)` misses and the
    # surviving head (endpoint+bucket+key) reaches LLM context.
    stderr = str((exec_res or {}).get("stderr") or "").replace(url, "<presigned-url>")[:_STDERR_SNIPPET_MAX]
    downloaded = _trailing_byte_count(str((exec_res or {}).get("stdout") or ""))

    if downloaded is None:
        logger.error("[ARCHIVE_TOOLS] fetch %s: no byte count in sandbox output", record["_id"])
        return _fetch_soft_error(
            f"archive_fetch: sandbox download produced no byte count: {stderr}",
            exec_res,
        )
    if downloaded == 0:
        return _fetch_soft_error(
            f"archive_fetch: sandbox download wrote no bytes: {stderr}",
            exec_res,
        )
    # Compare against the object's OWN length, not the database's. The stored
    # size is an advisory index written when the result was archived and can
    # disagree with the bytes actually in the bucket; trusting it would reject a
    # perfectly good download as short, and a retry would fail the same way.
    expected = head.get("ContentLength") if isinstance(head, dict) else None
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        expected = size_bytes if isinstance(size_bytes, int) else None
    if expected is not None and downloaded != expected:
        # A short file is the shape a killed or timed-out curl leaves behind, and
        # it is indistinguishable from a complete one by every other signal.
        return _fetch_soft_error(
            (
                f"archive_fetch: incomplete download — {downloaded} of {expected} bytes. "
                f"Retry archive_fetch, or use archive_query to search without downloading. {stderr}"
            ),
            exec_res,
        )

    note = (
        f"Ready: the full result is a file in your sandbox at {dest} ({downloaded} bytes). "
        "Process it there with whatever shell tooling you have — jq, awk, python, split, wc. "
        "Do not read the whole file back into the conversation; that is the context bloat "
        "this file exists to avoid. To search it without a download, archive_query runs "
        "server-side and returns matching lines."
    )
    if representation == "raw":
        # The one non-obvious property of the file: raw is json.dumps with no
        # indent, so head/tail/line-oriented tools all see a single line.
        note += (
            " This is the raw representation — one single line of JSON, so line-oriented "
            "tools are useless on it: use jq, or archive_fetch(representation='normalized') "
            "for the line-oriented copy."
        )

    # One size field only: `bytes` is what is actually on disk. The record's
    # advisory size would be a second, near-identical number the agent has to
    # choose between, and on success they are equal by definition.
    return {
        "status": "success",
        "ref_id": record["_id"],
        "representation": representation,
        "path": dest,
        "bytes": downloaded,
        "note": note,
    }
