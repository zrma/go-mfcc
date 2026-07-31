#!/usr/bin/env python3
"""Verify repository-owned inputs and generated outputs without the framework checkout."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath


ACTIVE_WORK_GLOBS = ("docs/todo-*/spec.md",)
STATUS_LINE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:상태|status)\s*:\s*(?P<value>.*?)\s*$",
    re.IGNORECASE,
)
STATUS_HEADING = re.compile(
    r"^\s{0,3}#{1,6}\s*(?:상태|status)\s*#*\s*$",
    re.IGNORECASE,
)
TERMINAL_STATUS_VALUE = re.compile(
    r"^(?:완료|complete(?:d)?|done|closed|implemented|superseded|deferred|"
    r"cancel(?:l)?ed)(?:\s|[.!:;—–-]|$)",
    re.IGNORECASE,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_relative(root: Path, value: str) -> Path:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or value in {"", "."}:
        raise ValueError(f"unsafe lock path: {value!r}")
    path = root.joinpath(*candidate.parts)
    resolved_root = root.resolve()
    resolved_path = path.resolve(strict=False)
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"lock path escapes repository: {value!r}")
    return path


def verify_group(root: Path, group: str, entries: object) -> list[str]:
    if not isinstance(entries, dict):
        return [f"{group} must be an object"]

    failures: list[str] = []
    for relative, expected in sorted(entries.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            failures.append(f"{group} contains a non-string entry")
            continue
        try:
            path = safe_relative(root, relative)
        except ValueError as error:
            failures.append(str(error))
            continue
        if not path.is_file():
            failures.append(f"missing {relative}")
        elif digest(path) != expected:
            failures.append(f"drifted {relative}")
    return failures


def verify_active_work(root: Path) -> list[str]:
    failures: list[str] = []
    for pattern in ACTIVE_WORK_GLOBS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError) as error:
                relative = path.relative_to(root).as_posix()
                failures.append(f"cannot read active-work packet {relative}: {error}")
                continue
            for line_number, line in enumerate(lines, start=1):
                normalized = (
                    line.replace("**", "").replace("__", "").replace("`", "")
                )
                status = STATUS_LINE.fullmatch(normalized)
                terminal_line = (
                    line_number
                    if status and TERMINAL_STATUS_VALUE.match(status.group("value"))
                    else None
                )
                if terminal_line is None and STATUS_HEADING.fullmatch(normalized):
                    for offset, value_line in enumerate(
                        lines[line_number:], start=line_number + 1
                    ):
                        normalized_value = (
                            value_line.replace("**", "")
                            .replace("__", "")
                            .replace("`", "")
                            .strip()
                        )
                        if not normalized_value:
                            continue
                        if TERMINAL_STATUS_VALUE.match(normalized_value):
                            terminal_line = offset
                        break
                if terminal_line is not None:
                    relative = path.relative_to(root).as_posix()
                    failures.append(
                        "completed active-work packet must be archived or removed: "
                        f"{relative}:{terminal_line}"
                    )
                    break
    return failures


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    lock_path = root / ".ai-first.lock"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"ai-first standalone check failed: invalid lock: {error}")
        return 1

    failures: list[str] = []
    if lock.get("schema_version") != 1:
        failures.append("unsupported lock schema")
    failures.extend(
        verify_group(root, "repository_inputs", lock.get("repository_inputs"))
    )
    failures.extend(verify_group(root, "outputs", lock.get("outputs")))
    failures.extend(verify_active_work(root))

    if failures:
        for failure in failures:
            print(f"ai-first standalone check failed: {failure}")
        return 1

    print("ai-first standalone check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
