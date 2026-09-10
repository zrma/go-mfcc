#!/usr/bin/env python3
"""Verify repository-owned inputs and generated outputs without the framework checkout."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path, PurePosixPath


ACTIVE_WORK_GLOBS = ("docs/todo-*/spec.md",)
AGENTS_HEADINGS = [
    "## First Read", "## AI-first Core Contract", "## Repository Overlay",
]
HARNESS_HEADINGS = [
    "## Interface", "## Project Objective", "## Core Operating Contract",
    "## Execution Loop", "## Verification And Evidence", "## Escalation",
    "## VCS And Publish", "## Harness Evaluation And Improvement",
    "## Capability Profiles", "## Project Overlay", "## Related Documents",
]
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
HEX_DIGEST = re.compile(r"[0-9a-f]{64}")
COMMIT_REVISION = re.compile(r"[0-9a-f]{40}")
RELEASE_REVISION = re.compile(
    r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_digest(entries: dict[str, str]) -> str:
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def safe_lock_key(value: str) -> bool:
    candidate = PurePosixPath(value)
    return not (
        candidate.is_absolute() or ".." in candidate.parts or value in {"", "."}
    )


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


def verify_lock_contract(root: Path, lock: dict[str, object]) -> list[str]:
    failures: list[str] = []
    config_path = root / ".ai-first.toml"
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        return [f"invalid .ai-first.toml: {error}"]

    framework = lock.get("framework")
    if not isinstance(framework, dict):
        return ["framework must be an object"]

    comparisons = (
        ("framework.version", framework.get("version"), config.get("framework_version")),
        (
            "framework.source_kind",
            framework.get("source_kind"),
            config.get("source_kind"),
        ),
        (
            "framework.source_revision",
            framework.get("source_revision"),
            config.get("source_revision"),
        ),
        ("profiles", lock.get("profiles"), config.get("profiles")),
    )
    for label, actual, expected in comparisons:
        if actual != expected:
            failures.append(f"{label} does not match .ai-first.toml")

    source_kind = framework.get("source_kind")
    source_revision = framework.get("source_revision")
    source_commit = framework.get("source_commit")
    if source_kind == "development":
        if source_revision is not None or source_commit is not None:
            failures.append(
                "development source requires null source_revision and source_commit"
            )
    elif source_kind == "commit":
        if not isinstance(source_revision, str) or not COMMIT_REVISION.fullmatch(
            source_revision
        ):
            failures.append("commit source_revision must be a full lowercase commit")
        if source_commit != source_revision:
            failures.append("commit source_commit must equal source_revision")
    elif source_kind == "release":
        if not isinstance(source_revision, str) or not RELEASE_REVISION.fullmatch(
            source_revision
        ):
            failures.append("release source_revision must be a stable version tag")
        if not isinstance(source_commit, str) or not COMMIT_REVISION.fullmatch(
            source_commit
        ):
            failures.append("release source_commit must be a full lowercase commit")
    else:
        failures.append("framework.source_kind is unsupported")

    framework_inputs = lock.get("framework_inputs")
    valid_inputs: dict[str, str] = {}
    if not isinstance(framework_inputs, dict) or not framework_inputs:
        failures.append("framework_inputs must be a non-empty object")
    else:
        for relative, expected in sorted(framework_inputs.items()):
            if not isinstance(relative, str) or not safe_lock_key(relative):
                failures.append("framework_inputs contains an unsafe path")
            elif not isinstance(expected, str) or not HEX_DIGEST.fullmatch(expected):
                failures.append(f"framework_inputs has invalid digest for {relative}")
            else:
                valid_inputs[relative] = expected
        if len(valid_inputs) == len(framework_inputs):
            expected_digest = aggregate_digest(valid_inputs)
            if framework.get("digest") != expected_digest:
                failures.append("framework.digest does not match framework_inputs")

    return failures


def config_path(value: object) -> str:
    # central parser와 같은 문자열/경로 정규화로 기존 선언의 동등 표기를 보존한다.
    if not isinstance(value, str) or not safe_lock_key(value.strip()):
        raise ValueError("invalid configured path")
    return PurePosixPath(value.strip()).as_posix()


def verify_interface(root: Path, lock: dict[str, object]) -> list[str]:
    """고정된 선언으로 공통 interface를 검증한다."""
    failures: list[str] = []
    try:
        config = tomllib.loads((root / ".ai-first.toml").read_text(encoding="utf-8"))
        output = config["output"]
        roles = {role: config_path(output[role]) for role in ("agents", "harness", "standalone_check")}
        if not all(isinstance(path, str) and safe_lock_key(path) for path in roles.values()):
            raise ValueError("invalid output role path")
        if len(set(roles.values())) != len(roles):
            raise ValueError("output roles must be distinct")
        entries = lock.get("outputs")
        if not isinstance(entries, dict) or not set(roles.values()) <= entries.keys():
            raise ValueError("lock is missing a configured output role")
        inputs = lock.get("repository_inputs")
        required_inputs = {".ai-first.toml", *(config_path(config["overlay"][role]) for role in ("agents_first_read", "agents_project", "harness_project"))}
        if not isinstance(inputs, dict) or not required_inputs <= inputs.keys():
            raise ValueError("lock is missing a configured repository input")
        agents = safe_relative(root, roles["agents"]).read_text(encoding="utf-8")
        harness = safe_relative(root, roles["harness"]).read_text(encoding="utf-8")
        profiles = config["profiles"]
        if not isinstance(profiles, list) or not profiles or not all(isinstance(p, str) for p in profiles):
            raise ValueError("profiles must be a non-empty string array")
        version = config["framework_version"]
        project = config["project"]
        lines = [
            "- Structure ID: `ai-first-harness-v1`.",
            f"- Framework version: `{version}`.",
            f"- Project: `{project['name'].strip()}`.",
            f"- Publication class: `{project['publication_class'].strip()}`.",
            f"- Generated drift check: `python3 {roles['standalone_check']}`.",
            f"- Publication boundary check: `{config_path(config['checks']['publication'])}`.",
        ]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError) as error:
        return [f"invalid shared interface: {error}"]

    header = (
        f"<!-- Generated by ai-first {version}. "
        "Edit .ai-first.toml or .ai-first/overlays/, then render. -->"
    )
    for label, content, headings in (
        ("agents", agents, AGENTS_HEADINGS), ("harness", harness, HARNESS_HEADINGS),
    ):
        if not content.startswith(header + "\n"):
            failures.append(f"{label} generated boundary does not match declaration")
        if [line for line in content.splitlines() if line.startswith("## ")] != headings:
            failures.append(f"{label} section order differs from ai-first-harness-v1")
    actual_profiles = [line for line in agents.splitlines() if line.startswith("### Capability Profile: ")]
    if actual_profiles != [f"### Capability Profile: {profile}" for profile in profiles]:
        failures.append("agents profile order does not match declaration")
    profile_section = harness.split("## Capability Profiles\n", 1)[-1].split("## Project Overlay\n", 1)[0]
    if [line for line in profile_section.splitlines() if line.startswith("### ")] != [f"### {profile}" for profile in profiles]:
        failures.append("harness profile order does not match declaration")
    for line in lines:
        if line not in harness.splitlines():
            failures.append("harness interface metadata does not match declaration: " + line.split(":", 1)[0])
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
                        "completed active-work packet requires artifact transfer and cleanup: "
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
    if not isinstance(lock, dict):
        print("ai-first standalone check failed: lock root must be an object")
        return 1

    failures = verify_lock_contract(root, lock)
    if lock.get("schema_version") != 1:
        failures.append("unsupported lock schema")
    failures.extend(
        verify_group(root, "repository_inputs", lock.get("repository_inputs"))
    )
    failures.extend(verify_group(root, "outputs", lock.get("outputs")))
    failures.extend(verify_interface(root, lock))
    failures.extend(verify_active_work(root))

    if failures:
        for failure in failures:
            print(f"ai-first standalone check failed: {failure}")
        return 1

    print("ai-first standalone check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
