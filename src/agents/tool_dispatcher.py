from __future__ import annotations

import json
import logging
import posixpath
import re
import shlex
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional, Set, Tuple

from llm.agent_model_params import DEFAULT_AGENT_MODEL
from tools.archive_tools import ARCHIVE_FETCH_DIR
from tools.attachment_tools import ATTACHMENT_FETCH_DIR
from tools.tenant_artifact_tools import TENANT_ARTIFACT_FETCH_DIR
from tools.unified_diff import UnifiedDiffError, apply_unified_diff
from tools.web_search import web_search
from deploy.stack_detector import detect_stack
from deploy.repair import analyze_and_repair
from storage.artifact_store import ArtifactStore
from schemas import TaskSchema
from context.shared_context import FULL_CONTEXT_READS_TOKEN
from .delegation import DelegationManager

logger = logging.getLogger(__name__)

# Bounds how long one search can take — the sandbox costs a round trip per
# directory listed and per file read — not what a search is allowed to look at.
_MAX_WALK_FILES = 5000


class SearchPathError(ValueError):
    """The search root could not be read at all, as opposed to being empty."""

    def __init__(
        self,
        message: str,
        *,
        error_type: Optional[str] = None,
        outcome_unknown: Any = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.outcome_unknown = outcome_unknown


def _search_path_error_from_tool(res: Dict[str, Any], fallback: str) -> SearchPathError:
    return SearchPathError(
        str(res.get("error") or fallback),
        error_type=res.get("error_type"),
        outcome_unknown=res.get("outcome_unknown"),
    )


def _search_tool_error(exc: SearchPathError, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": "error", "error": str(exc), "matches": [], **extra}
    if exc.error_type:
        out["error_type"] = exc.error_type
    if exc.outcome_unknown:
        out["outcome_unknown"] = exc.outcome_unknown
    return out


def _targets_archive_dir(command: str) -> bool:
    """Whether the command names a file under the archive-download dir.

    Deliberately not a parse of the command line. The redirect this exempts is a
    routing preference, not a boundary — `awk`, `sed`, `cat` and even `egrep`
    read the same files without passing through it — so the only job here is to
    stop it refusing the one search that HAS to happen in bash: a fetched
    archive, which lands under /tmp rather than the workdir the indexed grep
    tool walks, and is big enough that a per-file read is the wrong tool anyway.

    So a line that names an archive path is allowed to run, whatever else it
    searches. Trying to prove the search touches nothing else meant emulating
    getopt, and each round of that broke a normal way of writing grep — most
    recently `grep -C 3 ERROR <archive>`, where `3` was mistaken for the pattern.
    """
    try:
        tokens = shlex.split(command, comments=True)
    except ValueError:
        tokens = command.split()  # unbalanced quotes: still let a real archive path through
    for token in tokens:
        if not token.startswith("/"):
            continue
        resolved = posixpath.normpath(token)
        if resolved == ARCHIVE_FETCH_DIR or resolved.startswith(ARCHIVE_FETCH_DIR + "/"):
            return True
        if resolved == ATTACHMENT_FETCH_DIR or resolved.startswith(ATTACHMENT_FETCH_DIR + "/"):
            return True
        if resolved == TENANT_ARTIFACT_FETCH_DIR or resolved.startswith(TENANT_ARTIFACT_FETCH_DIR + "/"):
            return True
    return False


def _forward_refusal(res: Any, path: str) -> Optional[Dict[str, Any]]:
    """Forward an executor error (tenant deny, container down) instead of letting
    a handler rebuild it as success-with-empty-content — an agent that sees ""
    where the refusal was retries blind (three live dev runs, 2026-08-03)."""
    if isinstance(res, dict) and res.get("status") == "error":
        out: Dict[str, Any] = {
            "status": "error",
            "path": path,
            "error": str(res.get("error") or "tool execution failed"),
        }
        if res.get("error_type"):
            out["error_type"] = res["error_type"]
        if res.get("outcome_unknown"):
            out["outcome_unknown"] = res["outcome_unknown"]
        return out
    return None


class AgentToolDispatcher:
    def __init__(self, agent: Any):
        self.agent = agent
        self._read_paths: Set[str] = set()
        self._todo_path = ".AppFactory/todos.json"
        self._archive_store = None  # built on first use; moves oversized tool results to object storage
        # Set by generic_agent to splice the archive retrieval tools into the
        # live runner's schema list the moment a spill mints the run's first ref.
        self.on_archive_ref = None

        # Dict-based dispatch map — extensible without if/elif chain
        self._dispatch_map: Dict[str, Any] = {
            "create": self._create,
            "read": self._read,
            "grep": self._grep,
            "glob": self._glob,
            "edit": self._edit,
            "bash": self._bash,
            "web_search": self._web_search,
            "todo_read": lambda args: self._todo_read(),
            "todo_write": self._todo_write,
            "task": self._task,
            "delegate_to_agent": self._delegate_to_agent,
            "context_read": self._context_read,
            "context_write": self._context_write,
            "deploy_from_artifacts": self._deploy_from_artifacts,
            "detect_stack": self._detect_stack,
            "analyze_and_repair": self._analyze_and_repair,
        }

    def register_handler(self, tool_name: str, handler):
        """Register a custom dispatch handler for a tool name."""
        self._dispatch_map[tool_name] = handler

    @property
    def archive_store(self):
        """Built on first use. Reads S3 settings from the environment; gets the
        database handle (for saving the small locator record) from the agent's
        long-lived executor."""
        if self._archive_store is None:
            from storage.archive_store import ArchiveStore
            db = getattr(getattr(self.agent, "mcp_executor", None), "storage", None)
            self._archive_store = ArchiveStore.from_env(db)
        return self._archive_store

    async def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a tool call, then move the result into object storage if it
        is too big to inline.

        The guard sits HERE, at the LLM-facing tool boundary — not down inside
        mcp_executor.execute_tool. `_read`/`_edit`/`_grep` read a file's full
        content via execute_tool and post-process it (truncate, apply a diff)
        before returning; a guard underneath them would hand `_edit` an empty
        placeholder and write it back, wiping the file. By the time control
        returns here the result is final, and a large result from an external
        tool reaches it through the fall-through in `_handle_inner`, so both the
        container and external paths are covered.
        """
        exact_runtime = getattr(self.agent, "_exact_tool_runtime", None)
        result = (
            await exact_runtime.execute(payload)
            if exact_runtime is not None
            else await self._handle_inner(payload)
        )
        tool_id = payload.get("tool") if isinstance(payload, dict) else None
        sc = getattr(self.agent, "shared_context", None)
        # Canonical tenant so later retrieval can scope archives per tenant.
        # Absent tenant → root (an archive key needs a concrete scope; base.py's
        # tool-search instead passes None through). "__default__" → "__root__"
        # per the migrate_default_tenant_to_root rename.
        tenant_id = getattr(sc, "tenant_id", None) or "__root__"
        if tenant_id == "__default__":
            tenant_id = "__root__"
        # Vision bytes must reach the VL follow-up. Spill a journal-safe copy so
        # archive never stores data URLs; always return the live result when
        # image_data_url is present (scrubbed spill would drop the follow-up).
        from agents.vision_tool_result import has_live_vision_bytes, journal_safe_tool_result

        live_vision = has_live_vision_bytes(result)
        spill_input = journal_safe_tool_result(result) if live_vision else result
        spilled = await self.archive_store.maybe_spill(
            spill_input,
            project_id=getattr(sc, "project_id", None),
            tool_id=tool_id,
            run_id=getattr(sc, "run_id", None),
            agent_id=getattr(self.agent, "agent_id", None),
            tenant_id=tenant_id,
            tool_call_id=payload.get("tool_call_id") if isinstance(payload, dict) else None,
        )
        if isinstance(spilled, dict) and spilled.get("kind") == "archive_ref":
            await self._after_archive_spill(spilled)
        return result if live_vision else spilled

    async def _after_archive_spill(self, placeholder: Dict[str, Any]) -> None:
        """archive.created + the conditional-attach hook. Best-effort by design:
        the blob and ref are already safe — a UI event or schema splice failing
        must not turn a successful tool call into an error."""
        sc = getattr(self.agent, "shared_context", None)
        project_id = getattr(sc, "project_id", None)
        emitter = getattr(self.agent, "event_emitter", None)
        if emitter is not None and project_id:
            try:
                await emitter.emit(
                    "archive.created",
                    getattr(sc, "run_id", None),
                    {
                        "project_id": project_id,
                        "ref_id": placeholder.get("ref_id"),
                        "size_bytes": placeholder.get("size_bytes"),
                        "content_type": placeholder.get("content_type"),
                        "agent_id": getattr(self.agent, "agent_id", None),
                    },
                )
            except Exception as exc:
                logger.warning(
                    "[ARCHIVE] archive.created emit failed for %s: %s",
                    placeholder.get("ref_id"), exc,
                )
        hook = self.on_archive_ref
        if callable(hook):
            try:
                hook()
            except Exception as exc:
                logger.warning("[ARCHIVE] on_archive_ref hook failed: %s", exc)

    async def _handle_inner(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = payload.get("tool")
        args = payload.get("args") or {}
        if name == "ask_human":
            # The answer arrives keyed by the journal's tool_call_id, so the
            # handler must know which record it parked on. The run comes too,
            # because providers reuse call ids across runs — without it the
            # handler cannot tell this question's record from an old one's.
            # Scoped to ask_human — other fall-through tools are external MCP
            # tools whose args forward to remote servers verbatim.
            args = {
                **args,
                "_tool_call_id": payload.get("tool_call_id"),
                "_run_id": getattr(getattr(self.agent, "shared_context", None), "run_id", None),
            }
        if hasattr(self.agent, "ensure_not_cancelled"):
            self.agent.ensure_not_cancelled(f"tool_dispatch:{name or 'unknown'}")

        handler = self._dispatch_map.get(name)
        if handler:
            if name in ("deploy_from_artifacts", "detect_stack", "analyze_and_repair"):
                logger.info(f"[DEPLOY] [DISPATCH] tool call: {name}, args_keys={list(args.keys()) if args else []}")
            return await handler(args)

        # Fallback: route to agent.execute_tool which reaches MCPToolExecutor.
        # This covers external MCP tools (source=mcp_server) whose names are not
        # in _dispatch_map but whose schemas were exposed to the LLM via tool_registry.
        logger.info(f"[DISPATCH] No local handler for '{name}', routing to agent.execute_tool")
        try:
            result = await self.agent.execute_tool(name, args)
            return result
        except Exception as exc:
            logger.error(f"[DISPATCH] agent.execute_tool('{name}') failed: {exc}")
            return {"status": "error", "error": str(exc)}

    async def _read(self, args: Dict[str, Any]) -> Dict[str, Any]:
        attachment_id = str(args.get("attachment_id") or "").strip()
        path = str(args.get("path") or "").strip()

        if attachment_id and not path:
            fetch = await self.agent.execute_tool("attachment_fetch", {"attachment_id": attachment_id})
            refusal = _forward_refusal(fetch, attachment_id)
            if refusal:
                refusal["attachment_id"] = attachment_id
                return refusal
            if not isinstance(fetch, dict) or fetch.get("status") != "success":
                err = fetch.get("error") if isinstance(fetch, dict) else "attachment_fetch failed"
                return {"status": "error", "attachment_id": attachment_id, "error": str(err)}
            path = str(fetch.get("path") or "")

        if not path:
            return {"status": "error", "error": "Missing path or attachment_id"}

        res = await self.agent.execute_tool("read_file", {"path": path})
        refusal = _forward_refusal(res, path)
        if refusal:
            if attachment_id:
                refusal["attachment_id"] = attachment_id
            return refusal
        if not isinstance(res, dict):
            out = {"status": "error", "path": path, "error": "read_file returned no result"}
            if attachment_id:
                out["attachment_id"] = attachment_id
            return out
        content = res.get("content", "")

        start_line = args.get("start_line")
        end_line = args.get("end_line")
        if isinstance(start_line, int) and isinstance(end_line, int) and start_line >= 1 and end_line >= start_line:
            lines = content.splitlines(keepends=False)
            sliced = lines[start_line - 1 : end_line]
            content = "\n".join(sliced) + ("\n" if content.endswith("\n") else "")

        truncated = False
        total_lines = None
        if not (isinstance(start_line, int) and isinstance(end_line, int)):
            max_lines = int(args.get("max_lines") or 800)
            if max_lines > 0:
                lines = content.splitlines(keepends=False)
                total_lines = len(lines)
                if total_lines > max_lines:
                    content = "\n".join(lines[:max_lines])
                    if content and content[-1] != "\n" and res.get("content", "").endswith("\n"):
                        content += "\n"
                    truncated = True
                    start_line = 1
                    end_line = max_lines

        self._read_paths.add(path)
        payload = {"status": "success", "path": path, "content": content}
        if attachment_id:
            payload["attachment_id"] = attachment_id
        if truncated:
            payload.update({
                "truncated": True,
                "total_lines": total_lines,
                "start_line": start_line,
                "end_line": end_line,
            })
        return payload

    async def _bash(self, args: Dict[str, Any]) -> Dict[str, Any]:
        command = str(args.get("command") or "")
        if not command.strip():
            return {"status": "error", "error": "Missing command"}

        lowered = command.lower()
        # The redirect exists to push REPO searches onto the indexed grep tool.
        # Archive downloads live outside the workdir, so that tool structurally
        # cannot reach them (its walk is workdir-rooted) — for those files bash
        # grep IS the intended path (archive_fetch's note says so). Exempt only
        # when the archive dir is a real path ARGUMENT: a substring test let a
        # repo-wide grep with the dir in a trailing comment slip the redirect.
        if re.search(r"\b(?:grep|rg|ripgrep)\b", lowered) and not _targets_archive_dir(command):
            return {"status": "error", "error": "Use the grep tool, not grep/rg in bash"}

        # Prevent noisy file creation via `cat` + redirect/heredoc (use create/edit instead).
        # Keep other heredoc use-cases (e.g. python - <<'PY') working.
        if re.search(r"\bcat\b", lowered) and (">" in command or "<<" in command):
            return {"status": "error", "error": "Use create/edit tools for file writes; avoid cat-based writes in bash"}

        params: Dict[str, Any] = {"command": command}
        if args.get("cwd"):
            params["cwd"] = str(args.get("cwd"))
        return await self.agent.execute_tool("run_command", params)

    async def _create(self, args: Dict[str, Any]) -> Dict[str, Any]:
        path = str(args.get("path") or "").strip()
        content = args.get("content")
        if not path or not isinstance(content, str):
            return {"status": "error", "error": "Missing path or content"}
        exists, refusal = await self._file_exists_or_refusal(path)
        if refusal:
            return refusal
        if exists:
            return {"status": "error", "error": "File already exists; read then edit instead of create"}
        res = await self.agent.execute_tool("create_file", {"path": path, "content": content})
        refusal = _forward_refusal(res, path)
        if refusal:
            return refusal
        self._read_paths.add(path)
        return {"status": "success", "path": path}

    async def _web_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or "").strip()
        max_results = args.get("max_results")
        if isinstance(max_results, int):
            return await web_search(query=query, max_results=max_results)
        return await web_search(query=query)

    async def _glob(self, args: Dict[str, Any]) -> Dict[str, Any]:
        pattern = str(args.get("pattern") or "").strip()
        root = str(args.get("path") or ".").strip() or "."
        max_results = int(args.get("max_results") or 50)
        if not pattern:
            return {"status": "error", "error": "Missing pattern", "matches": []}

        try:
            paths = await self._walk_files(root)
        except SearchPathError as e:
            return _search_tool_error(e)

        matches: List[str] = []
        for p in paths:
            if fnmatch(p, pattern) or fnmatch(p.lstrip("./"), pattern):
                matches.append(p)
                if len(matches) >= max_results:
                    break
        return {"status": "success", "pattern": pattern, "matches": matches}

    async def _grep(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or "").strip()
        root = str(args.get("path") or ".").strip() or "."
        case_sensitive = bool(args.get("case_sensitive") or False)
        max_results = int(args.get("max_results") or 50)
        if not query:
            return {"status": "error", "error": "Missing query", "matches": []}

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            rx = re.compile(query, flags)
        except Exception as e:
            return {"status": "error", "error": f"Invalid regex: {e}", "matches": []}

        try:
            paths = await self._walk_files(root)
        except SearchPathError as e:
            return _search_tool_error(e, query=query)

        matches: List[Dict[str, Any]] = []
        for p in paths:
            rf = await self.agent.execute_tool("read_file", {"path": p})
            refusal = _forward_refusal(rf, p)
            if refusal:
                refusal["query"] = query
                refusal["matches"] = []
                return refusal
            content = (rf or {}).get("content", "") if isinstance(rf, dict) else ""
            if not content:
                continue
            for ln, line in enumerate(content.splitlines(keepends=False), start=1):
                if rx.search(line):
                    matches.append({"path": p, "line": ln, "text": line})
                    if len(matches) >= max_results:
                        return {"status": "success", "query": query, "matches": matches}
        return {"status": "success", "query": query, "matches": matches}

    async def _edit(self, args: Dict[str, Any]) -> Dict[str, Any]:
        path = str(args.get("path") or "").strip()
        diff_text = str(args.get("diff") or "")
        if not path or not diff_text:
            return {"status": "error", "error": "Missing path or diff"}

        exists, refusal = await self._file_exists_or_refusal(path)
        if refusal:
            return refusal
        if exists and path not in self._read_paths:
            return {"status": "error", "error": "Read before Edit: file has not been read"}

        original = ""
        if exists:
            rf = await self.agent.execute_tool("read_file", {"path": path})
            # A failed re-read must stop the edit: applying the diff against ""
            # writes the hunk alone back — truncating the file to that hunk.
            refusal = _forward_refusal(rf, path)
            if refusal is None and not isinstance(rf, dict):
                refusal = {"status": "error", "path": path, "error": "read_file returned no result"}
            if refusal:
                return refusal
            original = rf.get("content", "")

        try:
            updated = apply_unified_diff(original, diff_text)
        except UnifiedDiffError as e:
            return {"status": "error", "error": str(e)}

        res = await self.agent.execute_tool("edit_file", {"path": path, "content": updated})
        refusal = _forward_refusal(res, path)
        if refusal:
            return refusal
        self._read_paths.add(path)
        return {"status": "success", "path": path}

    async def _todo_read(self) -> Dict[str, Any]:
        exists, refusal = await self._file_exists_or_refusal(self._todo_path)
        if refusal:
            return refusal
        if not exists:
            return {"status": "success", "todos": []}
        rf = await self.agent.execute_tool("read_file", {"path": self._todo_path})
        refusal = _forward_refusal(rf, self._todo_path)
        if refusal:
            return refusal
        raw = (rf or {}).get("content", "") if isinstance(rf, dict) else ""
        try:
            data = json.loads(raw) if raw else []
        except Exception:
            data = []
        if not isinstance(data, list):
            data = []
        return {"status": "success", "todos": data}

    async def _todo_write(self, args: Dict[str, Any]) -> Dict[str, Any]:
        todos = args.get("todos")
        if not isinstance(todos, list):
            return {"status": "error", "error": "todos must be a list"}
        try:
            mkdir_res = await self.agent.execute_tool(
                "run_command", {"command": "mkdir -p .AppFactory", "cwd": "/workdir"}
            )
        except Exception as exc:
            return {
                "status": "error",
                "path": self._todo_path,
                "error": str(exc) or f"{type(exc).__name__} (no message)",
                "error_type": "unavailable",
            }
        # Typed infra on mkdir must reach the gate; do not mask with a later edit success.
        if isinstance(mkdir_res, dict) and (
            mkdir_res.get("error_type") or mkdir_res.get("outcome_unknown")
        ):
            out: Dict[str, Any] = {
                "status": "error",
                "path": self._todo_path,
                "error": str(
                    mkdir_res.get("error")
                    or mkdir_res.get("stderr")
                    or "mkdir .AppFactory failed"
                ),
            }
            if mkdir_res.get("error_type"):
                out["error_type"] = mkdir_res["error_type"]
            if mkdir_res.get("outcome_unknown"):
                out["outcome_unknown"] = mkdir_res["outcome_unknown"]
            return out
        payload = json.dumps(todos, ensure_ascii=False, indent=2)
        res = await self.agent.execute_tool(
            "edit_file", {"path": self._todo_path, "content": payload + "\n"}
        )
        refusal = _forward_refusal(res, self._todo_path)
        if refusal:
            return refusal
        return {"status": "success", "path": self._todo_path}

    async def _task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning(
            "[DISPATCH] tool='task' is deprecated; use 'delegate_to_agent' for full sub-agent capabilities"
        )
        role = str(args.get("role") or "").strip()
        prompt = str(args.get("prompt") or "").strip()
        if not role or not prompt:
            return {"status": "error", "error": "Missing role or prompt"}

        sys_map = {
            "researcher": "You are a focused researcher. Return concise findings and sources.",
            "doc_reader": "You read documentation and extract precise, actionable details.",
            "test_runner": "You propose minimal test commands and interpret failures.",
            "code_reviewer": "You review code changes for correctness and edge cases.",
        }
        sys = sys_map.get(role, "You are a helpful specialized worker.")

        llm = getattr(self.agent, "llm_client", None)
        if not llm:
            return {"status": "error", "error": "LLM client not available"}

        try:
            resp = await self.agent.call_llm(
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": prompt},
                ],
                model=getattr(self.agent, "model", None) or DEFAULT_AGENT_MODEL,
                temperature=0.2,
            )
        except Exception as e:
            return {"status": "error", "error": str(e)}

        content = resp if isinstance(resp, str) else (resp or {}).get("content", "")
        return {"status": "success", "content": content}

    async def _delegate_to_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return await DelegationManager(self.agent).delegate_to_agent(args)

    async def _context_read(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Read a logical SharedContext key for the current project."""
        key = str(args.get("key") or "").strip()
        if not key:
            return {"status": "error", "error": "Missing key"}

        self._warn_if_context_read_outside_contract(key)
        shared_context = getattr(self.agent, "shared_context", None)
        if not shared_context:
            return {"status": "error", "error": "shared_context not available"}

        reader = getattr(shared_context, "read_context_key_async", None)
        fallback_reader = getattr(shared_context, "read_context_key", None)
        if callable(reader):
            missing = object()
            value = await reader(key, default=missing)
        elif callable(fallback_reader):
            missing = object()
            value = fallback_reader(key, default=missing)
        else:
            return {"status": "error", "error": "shared_context does not support logical reads"}

        found = value is not missing
        return {
            "status": "success",
            "key": key,
            "found": found,
            "value": None if not found else value,
        }

    async def _context_write(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Write a logical SharedContext key for the current project."""
        key = str(args.get("key") or "").strip()
        if not key:
            return {"status": "error", "error": "Missing key"}
        if "value" not in args:
            return {"status": "error", "error": "Missing value"}

        self._warn_if_context_write_outside_contract(key)
        shared_context = getattr(self.agent, "shared_context", None)
        if not shared_context:
            return {"status": "error", "error": "shared_context not available"}

        writer = getattr(shared_context, "write_context_key", None)
        if not callable(writer):
            return {"status": "error", "error": "shared_context does not support logical writes"}

        try:
            await writer(key, args.get("value"))
        except Exception as exc:
            logger.warning(
                "[CONTEXT_TOOL] write_failed agent=%s key=%s error=%s",
                getattr(self.agent, "agent_id", "unknown"),
                key,
                exc,
            )
            return {"status": "error", "key": key, "error": str(exc)}

        return {"status": "success", "key": key, "value": args.get("value")}

    def _warn_if_context_read_outside_contract(self, key: str) -> None:
        """Log soft contract drift without blocking dynamic context access."""
        task = getattr(self.agent, "current_task", None)
        if not isinstance(task, dict) or not TaskSchema.has_reads(task):
            return
        reads = TaskSchema.get_reads(task) or []
        if FULL_CONTEXT_READS_TOKEN in reads or key in reads:
            return
        logger.warning(
            "[CONTEXT_TOOL] agent=%s key=%s declared_reads=%s - read outside declared reads",
            getattr(self.agent, "agent_id", "unknown"),
            key,
            reads,
        )

    def _warn_if_context_write_outside_contract(self, key: str) -> None:
        """Log soft contract drift without blocking dynamic context writes."""
        task = getattr(self.agent, "current_task", None)
        if not isinstance(task, dict) or not TaskSchema.has_writes(task):
            return
        writes = TaskSchema.get_writes(task) or []
        if key in writes:
            return
        logger.warning(
            "[CONTEXT_TOOL] agent=%s key=%s declared_writes=%s - write outside declared writes",
            getattr(self.agent, "agent_id", "unknown"),
            key,
            writes,
        )

    async def _deploy_from_artifacts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call deploy_service.deploy_from_artifacts_prod; used by deploy GenericAgent."""
        logger.info(f"[DEPLOY] [DISPATCH] _deploy_from_artifacts called: deploy_slug={args.get('deploy_slug')}, has_deploy_spec={args.get('deploy_spec') is not None}")
        deploy_service = getattr(self.agent, "deploy_service", None)
        shared_context = getattr(self.agent, "shared_context", None)
        if not deploy_service or not shared_context:
            logger.warning(f"[DEPLOY] [DISPATCH] deploy_from_artifacts: deploy_service or shared_context missing")
            return {"status": "error", "error": "deploy_service or shared_context not available"}
        project_id = getattr(shared_context, "project_id", None) or "unknown"
        deploy_slug = args.get("deploy_slug")
        if not deploy_slug:
            logger.warning(f"[DEPLOY] [DISPATCH] deploy_from_artifacts: Missing deploy_slug")
            return {"status": "error", "error": "Missing deploy_slug"}
        target_namespace = args.get("target_namespace") or "AppFactory-apps"
        deploy_spec = args.get("deploy_spec")
        try:
            result = await deploy_service.deploy_from_artifacts_prod(
                project_id=project_id,
                shared_context=shared_context,
                deploy_slug=deploy_slug,
                target_namespace=target_namespace,
                deploy_spec=deploy_spec,
            )
            logger.info(f"[DEPLOY] [DISPATCH] _deploy_from_artifacts done: project_id={project_id}, result_keys={list(result.keys()) if isinstance(result, dict) else type(result).__name__}")
            # Normalize result: always produce stable deploy_status + error so
            # the orchestrator gets a predictable contract regardless of how
            # deploy_service.deploy_from_artifacts_prod responds.
            status_str = result.get("deploy_status") if isinstance(result, dict) else None
            deployment = (result.get("deployment") if isinstance(result, dict) else None) or {}
            image_ref = deployment.get("image_ref")
            deploy_status = "succeeded" if status_str == "succeeded" else "failed"
            err_obj = result.get("error") if isinstance(result, dict) else None
            error = (err_obj.get("message") or err_obj.get("error") or "") if isinstance(err_obj, dict) else (err_obj or "")

            return {
                "status": "success",
                "deploy_status": deploy_status,
                "error": error,
                "image_ref": image_ref,
                "result": result,
            }
        except Exception as e:
            logger.exception(f"[DEPLOY] [DISPATCH] deploy_from_artifacts failed: {e}")
            return {
                "status": "error",
                "deploy_status": "failed",
                "error": str(e),
                "image_ref": None,
            }

    async def _detect_stack(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Detect project stack from ArtifactStore artifacts; used by deploy GenericAgent."""
        logger.info(f"[DEPLOY] [DISPATCH] _detect_stack called")
        shared_context = getattr(self.agent, "shared_context", None)
        if not shared_context:
            logger.warning(f"[DEPLOY] [DISPATCH] _detect_stack: shared_context not available")
            return {"status": "error", "error": "shared_context not available"}
        project_id = getattr(shared_context, "project_id", None)
        if not isinstance(project_id, str) or not project_id.strip():
            logger.warning("[DEPLOY] [DISPATCH] _detect_stack: shared_context.project_id missing")
            return {"status": "error", "error": "shared_context.project_id missing"}
        file_artifacts = await self._load_artifacts_from_store(project_id)
        # Fallback: when ArtifactStore is empty or unavailable, inspect representative files
        # from the current workdir so deploy tooling can still make progress locally.
        if not file_artifacts:
            try:
                candidates = [
                    "package.json",
                    "pnpm-lock.yaml",
                    "yarn.lock",
                    "package-lock.json",
                    "pyproject.toml",
                    "requirements.txt",
                    "Pipfile",
                    "poetry.lock",
                    "setup.py",
                    "go.mod",
                    "Cargo.toml",
                    "pom.xml",
                    "build.gradle",
                    "build.gradle.kts",
                    "composer.json",
                    "Dockerfile",
                ]
                found: List[str] = []

                # Prefer repo root matches first
                for name in candidates:
                    if await self._file_exists(name) is True:
                        found.append(name)
                        if len(found) >= 8:
                            break

                # If nothing at root, scan for common subpaths (frontend/, backend/, app/, src/)
                if not found:
                    wanted = set(candidates)
                    for p in await self._walk_files("."):
                        base = p.replace("\\", "/").rsplit("/", 1)[-1]
                        if base in wanted:
                            found.append(p)
                            if len(found) >= 8:
                                break

                for p in found:
                    rf = await self.agent.execute_tool("read_file", {"path": p})
                    content = (rf or {}).get("content") if isinstance(rf, dict) else None
                    if not isinstance(content, str) or not content:
                        continue
                    # Prevent huge files from overwhelming detection
                    if len(content) > 200_000:
                        content = content[:200_000]
                    file_artifacts.append({"path": p, "content": content})

                if not file_artifacts:
                    logger.warning(
                        f"[DEPLOY] [DISPATCH] _detect_stack: no artifacts in ArtifactStore and no representative files found in workdir"
                    )
            except Exception as e:
                logger.warning(f"[DEPLOY] [DISPATCH] _detect_stack fallback failed: {e}")
        detection = detect_stack(file_artifacts)
        logger.info(f"[DEPLOY] [DISPATCH] _detect_stack done: stack={getattr(detection, 'stack', None)}, details_keys={list(detection.details.keys()) if getattr(detection, 'details', None) else None}")
        return {"status": "success", "stack": detection.stack, "details": detection.details}

    async def _analyze_and_repair(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run repair logic on deploy failure diagnostics; used by deploy GenericAgent."""
        attempt = int(args.get("attempt", 1)) if args.get("attempt") is not None else 1
        logger.info(f"[DEPLOY] [DISPATCH] _analyze_and_repair called: attempt={attempt}, has_diagnostics={args.get('diagnostics') is not None}")
        llm_client = getattr(self.agent, "llm_client", None)
        shared_context = getattr(self.agent, "shared_context", None)
        diagnostics = args.get("diagnostics")
        if not isinstance(diagnostics, dict):
            logger.warning(f"[DEPLOY] [DISPATCH] _analyze_and_repair: Missing or invalid diagnostics")
            return {"status": "error", "error": "Missing or invalid diagnostics"}
        if not llm_client or not shared_context:
            logger.warning(f"[DEPLOY] [DISPATCH] _analyze_and_repair: llm_client or shared_context not available")
            return {"status": "error", "error": "llm_client or shared_context not available"}
        storage = self._resolve_storage()
        try:
            applied, needs_delegation = await analyze_and_repair(
                llm_client,
                shared_context,
                diagnostics,
                attempt,
                storage=storage,
            )
            logger.info(f"[DEPLOY] [DISPATCH] _analyze_and_repair done: applied={applied}, needs_delegation={needs_delegation}")
            return {
                "status": "success",
                "applied": applied,
                "needs_delegation": needs_delegation,
            }
        except Exception as e:
            logger.exception(f"[DEPLOY] [DISPATCH] analyze_and_repair failed: {e}")
            return {"status": "error", "error": str(e)}

    def _resolve_storage(self):
        """Resolve the storage backend from the agent or its deploy service."""
        storage = getattr(self.agent, "storage", None)
        if storage is not None:
            return storage
        deploy_service = getattr(self.agent, "deploy_service", None)
        return getattr(deploy_service, "storage", None)

    async def _load_artifacts_from_store(self, project_id: str) -> List[Dict[str, Any]]:
        """Load file artifacts for deploy tools from ArtifactStore."""
        storage = self._resolve_storage()
        if storage is None:
            logger.warning(
                "[DEPLOY] [DISPATCH] project_id=%s storage_available=false - ArtifactStore unavailable",
                project_id,
            )
            return []

        try:
            artifact_store = ArtifactStore(storage)
            await artifact_store.initialize()
        except Exception as exc:
            logger.warning(
                "[DEPLOY] [DISPATCH] project_id=%s storage_init_failed=true error=%s - failed to initialize ArtifactStore",
                project_id,
                exc,
            )
            return []
        if artifact_store.collection is None:
            logger.warning(
                "[DEPLOY] [DISPATCH] project_id=%s artifact_store_ready=false - file_artifacts collection unavailable",
                project_id,
            )
            return []

        try:
            artifacts = await artifact_store.get_all_files(project_id)
        except Exception as exc:
            logger.warning(
                "[DEPLOY] [DISPATCH] project_id=%s load_failed=true error=%s - failed to load file artifacts",
                project_id,
                exc,
            )
            return []
        file_artifacts: List[Dict[str, Any]] = []
        for artifact in artifacts:
            path = (artifact.get("path") or "").strip()
            content = artifact.get("content")
            if not path or not isinstance(content, str):
                continue
            file_artifacts.append({"path": path, "content": content})

        logger.info(
            "[DEPLOY] [DISPATCH] project_id=%s artifacts=%d - loaded artifacts from ArtifactStore",
            project_id,
            len(file_artifacts),
        )
        return file_artifacts

    async def _file_exists_or_refusal(
        self, path: str
    ) -> Tuple[Optional[bool], Optional[Dict[str, Any]]]:
        """(exists, None) when listing answered; (None, error) only on typed infra.

        Domain listing misses (missing parent dir, SandboxListingError without
        error_type) are exists=False — same soft-fail as create_file — so nested
        create and first todo_read still work. Typed infra must still refuse.
        """
        norm = path.strip().lstrip("./")
        parent, name = _split_parent(norm)
        lf = await self.agent.execute_tool("list_files", {"path": parent or "."})
        if not isinstance(lf, dict):
            return None, {
                "status": "error",
                "path": path,
                "error": "could not verify whether the file exists (listing failed)",
                "outcome_unknown": True,
            }
        if lf.get("error_type") or lf.get("outcome_unknown"):
            out: Dict[str, Any] = {
                "status": "error",
                "path": path,
                "error": str(
                    lf.get("error")
                    or "could not verify whether the file exists (listing failed)"
                ),
            }
            if lf.get("error_type"):
                out["error_type"] = lf["error_type"]
            if lf.get("outcome_unknown"):
                out["outcome_unknown"] = lf["outcome_unknown"]
            return None, out
        files = lf.get("files", [])
        for entry in files or []:
            if not isinstance(entry, str):
                continue
            s = entry.rstrip("/")
            base = s.rsplit("/", 1)[-1]
            if base == name:
                return True, None
        return False, None

    async def _file_exists(self, path: str) -> Optional[bool]:
        exists, _ = await self._file_exists_or_refusal(path)
        return exists

    async def _walk_files(self, root: str) -> List[str]:
        start = root or "."
        queue: List[str] = [start]
        visited: Set[str] = set()
        out: List[str] = []

        while queue and len(out) < _MAX_WALK_FILES:
            d = queue.pop(0)
            if d in visited:
                continue
            visited.add(d)

            lf = await self.agent.execute_tool("list_files", {"path": d})
            listing_failed = isinstance(lf, dict) and (
                lf.get("status") == "error"
                or lf.get("error_type")
                or lf.get("outcome_unknown")
            )
            if listing_failed:
                is_infra = bool(lf.get("error_type") or lf.get("outcome_unknown"))
                # Mid-walk domain failures (file mistaken for a directory) skip
                # that branch; typed infra must not look like an empty tree.
                if d != start:
                    if is_infra:
                        raise _search_path_error_from_tool(lf, f"could not list {d}")
                    continue
                if is_infra:
                    raise _search_path_error_from_tool(lf, f"could not list {d}")
                # The root is not a directory. Reading one named file is the
                # obvious thing to ask for, so read it and search that; only a
                # root we cannot open at all is worth refusing over. An
                # unreadable path reads back empty — the sandbox error is not
                # handed back as content — so empty content here means refuse.
                rf = await self.agent.execute_tool("read_file", {"path": d})
                if isinstance(rf, dict) and (
                    rf.get("status") == "error"
                    or rf.get("error_type")
                    or rf.get("outcome_unknown")
                ):
                    raise _search_path_error_from_tool(
                        rf, str(lf.get("error") or f"could not read {d}")
                    )
                if isinstance(rf, dict) and rf.get("content"):
                    return [d]
                raise _search_path_error_from_tool(lf, f"could not read {d}")
            entries = (lf or {}).get("files", []) if isinstance(lf, dict) else []

            for entry in entries or []:
                if not isinstance(entry, str) or not entry.strip("/ "):
                    continue
                p = _join(d, entry)
                low = p.lower()

                # Matched on any segment, not just the root: a nested .git or
                # vendored node_modules used to be filtered out downstream by
                # the extension whitelist, and nothing filters it now.
                if "/.git/" in low or low.startswith(".git/"):
                    continue
                if "/node_modules/" in low or low.startswith("node_modules/"):
                    continue
                if "/__pycache__/" in low or low.startswith("__pycache__/"):
                    continue

                # The listing marks a directory with a trailing slash. The old
                # test was "does the name contain a dot", which sent Makefile,
                # LICENSE and every extensionless file down the directory
                # branch — a failing round trip each, then silently dropped.
                if p.endswith("/"):
                    child = p.rstrip("/")
                    if child and child not in visited:
                        queue.append(child)
                    continue

                out.append(p)
                if len(out) >= _MAX_WALK_FILES:
                    break

        return out


def _join(dir_path: str, entry: str) -> str:
    d = (dir_path or ".").rstrip("/")
    e = entry.lstrip("/")
    if d in ("", "."):
        return e
    if e.startswith(d + "/"):
        return e
    return f"{d}/{e}"


def _split_parent(path: str) -> Tuple[str, str]:
    p = path.strip().rstrip("/")
    if "/" not in p:
        return ".", p
    parent, name = p.rsplit("/", 1)
    return parent or ".", name
