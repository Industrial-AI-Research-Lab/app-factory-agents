from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str]


class UnifiedDiffError(ValueError):
    pass


def apply_unified_diff(original: str, diff_text: str) -> str:
    original_lines = original.splitlines(keepends=False)
    hunks = _parse_hunks(diff_text)

    out: List[str] = []
    src_i = 0

    for h in hunks:
        # unified diff line numbers are 1-based
        target_src_i = max(0, h.old_start - 1)
        if target_src_i < src_i:
            raise UnifiedDiffError("Overlapping hunks")

        out.extend(original_lines[src_i:target_src_i])
        src_i = target_src_i

        for dl in h.lines:
            if not dl:
                raise UnifiedDiffError("Empty diff line")
            prefix = dl[0]
            text = dl[1:]

            if prefix == " ":
                if src_i >= len(original_lines) or original_lines[src_i] != text:
                    raise UnifiedDiffError("Context mismatch")
                out.append(text)
                src_i += 1
            elif prefix == "-":
                if src_i >= len(original_lines) or original_lines[src_i] != text:
                    raise UnifiedDiffError("Delete mismatch")
                src_i += 1
            elif prefix == "+":
                out.append(text)
            elif prefix == "\\":
                # "\\ No newline at end of file"; ignore
                continue
            else:
                raise UnifiedDiffError(f"Unexpected diff line prefix: {prefix!r}")

    out.extend(original_lines[src_i:])
    return "\n".join(out) + ("\n" if original.endswith("\n") else "")


def _parse_hunks(diff_text: str) -> List[Hunk]:
    lines = diff_text.splitlines(keepends=False)
    hunks: List[Hunk] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith("@@"):
            header = line
            old_start, old_count, new_start, new_count = _parse_hunk_header(header)
            i += 1
            hunk_lines: List[str] = []
            while i < len(lines):
                diff_line = lines[i]
                if diff_line.startswith("@@"):
                    break
                if diff_line.startswith("--- ") or diff_line.startswith("+++ "):
                    # file headers may appear before hunks; treat as boundary
                    break
                hunk_lines.append(diff_line)
                i += 1
            hunks.append(
                Hunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    lines=hunk_lines,
                )
            )
            continue
        i += 1

    if not hunks:
        raise UnifiedDiffError("No hunks found")
    return hunks


def _parse_hunk_header(header: str) -> Tuple[int, int, int, int]:
    # @@ -l,s +l,s @@
    try:
        mid = header.split("@@", 2)[1].strip()
        parts = mid.split(" ")
        old_part = next(p for p in parts if p.startswith("-"))
        new_part = next(p for p in parts if p.startswith("+"))

        old_start, old_count = _parse_range(old_part[1:])
        new_start, new_count = _parse_range(new_part[1:])
        return old_start, old_count, new_start, new_count
    except Exception as e:
        raise UnifiedDiffError(f"Invalid hunk header: {header!r}") from e


def _parse_range(r: str) -> Tuple[int, int]:
    if "," in r:
        a, b = r.split(",", 1)
        return int(a), int(b)
    return int(r), 1
