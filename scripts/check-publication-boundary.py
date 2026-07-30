#!/usr/bin/env python3
"""Reject machine-local and cross-repository details from public artifacts."""

from __future__ import annotations

import argparse
import ipaddress
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CHECKER_PATH = "scripts/check-publication-boundary.py"
SAFE_HOME_USERS = {"example", "local-user", "me", "runner", "tester", "user", "you"}
DOCUMENTATION_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")
)


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    kind: str


def run(root: Path, *command: str) -> str:
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        raise RuntimeError(detail[-1] if detail else f"{command[0]} failed")
    return completed.stdout


def root() -> Path:
    return Path(run(Path.cwd(), "git", "rev-parse", "--show-toplevel").strip())


def repository_identity(repo: Path) -> tuple[str, str]:
    remote = run(repo, "git", "config", "--get", "remote.origin.url").strip()
    match = re.search(r"(?:github\.com[/:])([^/]+)/([^/#]+?)(?:\.git)?$", remote)
    if not match:
        raise RuntimeError("origin does not identify a GitHub repository")
    return match.group(1), match.group(2)


def files(repo: Path) -> Iterable[tuple[str, str]]:
    values = {
        value
        for value in run(repo, "git", "ls-files", "-z").split("\0")
        if value
    }
    values.update(
        value
        for value in run(
            repo,
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ).split("\0")
        if value
    )
    for relative in sorted(values):
        if relative == CHECKER_PATH:
            continue
        path = repo / relative
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        yield relative, data.decode("utf-8", errors="ignore")


def scan_text(
    relative: str,
    text: str,
    owner: str,
    repository: str,
) -> set[Finding]:
    findings: set[Finding] = set()
    home = re.compile(
        r"(?<![A-Za-z0-9_.-])/" + r"(?:Users|home)/([A-Za-z0-9._-]+)"
    )
    windows_home = re.compile(
        r"(?i)(?<![A-Za-z0-9_.-])[A-Z]:"
        + re.escape("\\")
        + r"Users"
        + re.escape("\\")
        + r"([A-Za-z0-9._-]+)"
    )
    private_hostname = re.compile(
        r"(?i)\b[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
        r"(?:\.[a-z0-9-]+)*\.(?:local|internal|lan|home\.arpa|ts\.net)\b"
    )
    ipv4 = re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
    sibling = re.compile(
        rf"(?i)(?:https?://github\.com/|git@github\.com:)"
        rf"{re.escape(owner)}/"
        rf"(?!{re.escape(repository)}(?:\.git)?(?:[/\s`'\"]|$))"
        r"[A-Za-z0-9_.-]+"
    )
    raw_evidence = re.compile(
        r"(?i)(?:healthcheck|diagnostic|support-bundle|cluster-dump)"
        r"[-_][0-9]{8}(?:[-_][0-9]{4,6})?"
    )
    record_like = Path(relative).suffix.lower() in {".log", ".md", ".txt"}

    for line_number, line in enumerate(text.splitlines(), start=1):
        for match in home.finditer(line):
            if match.group(1).lower() not in SAFE_HOME_USERS:
                findings.add(
                    Finding(relative, line_number, "machine-local-home-path")
                )
        for match in windows_home.finditer(line):
            if match.group(1).lower() not in SAFE_HOME_USERS:
                findings.add(
                    Finding(relative, line_number, "machine-local-home-path")
                )
        if record_like and private_hostname.search(line):
            findings.add(Finding(relative, line_number, "private-hostname"))
        if sibling.search(line):
            findings.add(Finding(relative, line_number, "sibling-repository"))
        if raw_evidence.search(line):
            findings.add(Finding(relative, line_number, "raw-runtime-evidence"))
        for match in ipv4.finditer(line) if record_like else ():
            try:
                address = ipaddress.ip_address(match.group(0))
            except ValueError:
                continue
            if (
                address.is_loopback
                or address.is_unspecified
                or any(address in network for network in DOCUMENTATION_NETWORKS)
            ):
                continue
            if address.is_private or not address.is_global:
                findings.add(
                    Finding(relative, line_number, "specific-network-address")
                )
    return findings


def publication_class(repo: Path) -> str:
    harness = (repo / "docs" / "agent-harness.md").read_text(encoding="utf-8")
    matches = re.findall(
        r"^- Publication class: `(public|internal)`\.$",
        harness,
        flags=re.MULTILINE,
    )
    if matches != ["public"]:
        raise RuntimeError("generated harness must declare class=public exactly once")
    expected = f"- Publication boundary check: `{CHECKER_PATH}`."
    if harness.count(expected) != 1:
        raise RuntimeError("generated harness does not declare the canonical checker")
    return matches[0]


def self_test() -> int:
    unsafe = (
        "built under " + "/" + "Users/private-user/project",
        "connect to node.private.internal",
        "source https://github.com/example/private-repo",
        "target 10.20.30.40",
    )
    safe = (
        "use <home>/<repo-root>",
        "source https://github.com/example/public-repo",
        "example 192.0.2.10",
    )
    for fixture in unsafe:
        if not scan_text("fixture.md", fixture, "example", "public-repo"):
            print("publication boundary self-test failed: unsafe fixture accepted")
            return 1
    for fixture in safe:
        if scan_text("fixture.md", fixture, "example", "public-repo"):
            print("publication boundary self-test failed: safe fixture rejected")
            return 1
    print("publication boundary self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    try:
        repo = root()
        publication_class(repo)
        visibility = os.environ.get(
            "PUBLICATION_LIVE_VISIBILITY",
            "",
        ).strip().lower()
        if visibility and visibility != "public":
            raise RuntimeError("declared class does not match live visibility")
        owner, repository = repository_identity(repo)
        findings: set[Finding] = set()
        for relative, text in files(repo):
            findings.update(scan_text(relative, text, owner, repository))
    except (OSError, RuntimeError, UnicodeError) as error:
        print(f"publication boundary check failed: {error}")
        return 1

    if findings:
        for finding in sorted(findings):
            print(
                "publication boundary finding: "
                f"path={finding.path} line={finding.line} class={finding.kind}"
            )
        print(f"publication boundary check failed: {len(findings)} finding(s)")
        return 1

    print("publication boundary check passed: class=public")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
