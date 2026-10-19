#!/usr/bin/env python3
"""main 이력과 명시적으로 종료하는 로컬 작업의 잔여물을 검사한다."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tomllib


class StateError(Exception):
    pass


def query(root: Path, *args: str, optional: bool = False) -> str:
    try:
        result = subprocess.run(args, cwd=root, capture_output=True, text=True)
    except FileNotFoundError as error:
        if optional:
            return ""
        raise StateError(f"required command is unavailable: {args[0]}") from error
    if result.returncode:
        if optional:
            return ""
        raise StateError(f"{args[0]} query failed; cannot verify repository state")
    return result.stdout.rstrip("\n")


def jj(root: Path, *args: str) -> str:
    return query(root, "jj", "--no-pager", "--ignore-working-copy", *args)


def load_policy(root: Path) -> tuple[str, str | None]:
    path = root / ".ai-first.toml"
    try:
        policy = tomllib.loads(path.read_text(encoding="utf-8")).get("vcs", {}) if path.exists() else {}
    except (OSError, ValueError) as error:
        raise StateError("cannot read VCS policy") from error
    if not isinstance(policy, dict):
        raise StateError("vcs must be a table")
    bookmark = policy.get("bookmark", "main")
    base = policy.get("linear_history_base")
    if not isinstance(bookmark, str) or not bookmark or any(c.isspace() for c in bookmark):
        raise StateError("vcs.bookmark must be a nonempty bookmark name")
    if base is not None and (not isinstance(base, str) or not re.fullmatch(r"[0-9a-f]{40}", base)):
        raise StateError("vcs.linear_history_base must be a full lowercase commit ID")
    return bookmark, base


def revision_id(root: Path, revision: str, native: bool) -> str:
    if native:
        commits = jj(root, "log", "-r", revision, "--no-graph", "-T", 'commit_id ++ "\n"').splitlines()
        if len(commits) != 1:
            raise StateError("revision must resolve to exactly one commit")
        return commits[0]
    return query(root, "git", "rev-parse", "--verify", f"{revision}^{{commit}}")


def check_history(root: Path, revision: str | None, base: str | None = None) -> tuple[str, bool]:
    bookmark, policy_base = load_policy(root)
    revision = revision or bookmark
    base = base or policy_base
    native = bool(query(root, "jj", "root", optional=True))
    if query(root, "git", "rev-parse", "--is-shallow-repository", optional=True) == "true":
        raise StateError("full history is required; use fetch-depth: 0")
    if native:
        target = revision_id(root, revision, True)
        checked = f"ancestors({target})"
        if base:
            base_id = revision_id(root, base, True)
            if not jj(root, "log", "-r", f"{base_id} & ancestors({target})", "--no-graph", "-T", "commit_id"):
                raise StateError("linear history base is not an ancestor of the candidate")
            checked = f"{base_id}..{target}"
        merges = jj(root, "log", "-r", f"({checked}) & merges()", "--no-graph", "-T", 'commit_id ++ "\n"')
    else:
        if query(root, "git", "rev-parse", "--is-shallow-repository") != "false":
            raise StateError("full history is required; use fetch-depth: 0")
        if revision == bookmark and not query(root, "git", "rev-parse", "--verify", f"refs/heads/{bookmark}", optional=True):
            revision = "origin/" + bookmark
        target = revision_id(root, revision, False)
        checked = target
        if base:
            base_id = revision_id(root, base, False)
            ancestors = query(root, "git", "rev-list", target).splitlines()
            if base_id not in ancestors:
                raise StateError("linear history base is not an ancestor of the candidate")
            checked = f"{base_id}..{target}"
        merges = query(root, "git", "rev-list", "--min-parents=2", checked)
    if merges:
        raise StateError("main history must be linear; merge commits were found")
    return target, native


def check_closeout(root: Path, target: str, keep: set[str], retired_paths: list[Path], inherited: str | None = None, keep_heads: list[str] = (), keep_worktrees: list[Path] = ()) -> None:
    query(root, "jj", "--no-pager", "status")
    if inherited:
        if not re.fullmatch(r"[0-9a-f]{40}", inherited) or jj(root, "log", "-r", "@", "--no-graph", "-T", "commit_id") != inherited:
            raise StateError("inherited working copy changed; preserve and classify it before closeout")
    else:
        if jj(root, "log", "-r", "@", "--no-graph", "-T", "empty") != "true":
            raise StateError("current working copy is not empty")
        parents = jj(root, "log", "-r", "parents(@)", "--no-graph", "-T", 'commit_id ++ "\n"').splitlines()
        if parents != [target]:
            raise StateError("current working copy must have exactly the expected parent")

    records = jj(root, "workspace", "list", "-T", r'name ++ "\0" ++ root ++ "\0" ++ target.commit_id() ++ "\n"')
    rows = [line.split("\0") for line in records.splitlines()]
    if any(len(row) != 3 for row in rows):
        raise StateError("workspace root metadata is missing")
    current = jj(root, "log", "-r", "@", "--no-graph", "-T", "commit_id")
    current_rows = [row for row in rows if row[2] == current]
    workspaces = {}
    for name, path, target_id in rows:
        # jj 0.38 이전 등록은 root가 없다. 현재 사본의 위치만 CLI와 유일한 WC ID로 검증한다.
        if not path and target_id == current and len(current_rows) == 1:
            path = jj(root, "root")
        if not path:
            raise StateError("workspace root metadata is missing")
        workspaces[name] = Path(path).resolve()
    if keep - workspaces.keys():
        raise StateError("a retained workspace does not exist")
    if workspaces.keys() - keep:
        raise StateError("unclassified secondary workspaces remain; retain active work explicitly")
    if any(not path.is_dir() for path in workspaces.values()):
        raise StateError("workspace registration points to a missing directory")
    if root.resolve() not in workspaces.values():
        raise StateError("current workspace is not registered")

    orphans = jj(root, "log", "-r", "heads(all()) ~ (bookmarks() | tags() | working_copies())", "--no-graph", "-T", 'commit_id ++ "\n"')
    retained_heads = {revision_id(root, revision, True) for revision in keep_heads}
    if set(orphans.splitlines()) - retained_heads:
        raise StateError("unreferenced heads remain; classify and preserve unique work before cleanup")

    # Git 등록, jj 등록, 실제 디렉터리는 서로 별개로 확인한다.
    if (root / ".git").exists() or (root / ".git").is_symlink():
        query(root, "git", "rev-parse", "--git-dir")
        records = query(root, "git", "worktree", "list", "--porcelain", "-z")
        paths = {
            Path(field.removeprefix("worktree ")).resolve()
            for record in records.split("\0\0")
            for field in record.split("\0")
            if field.startswith("worktree ")
        }
        if paths - set(workspaces.values()) - {p.resolve() for p in keep_worktrees}:
            raise StateError("unclassified Git worktree registrations remain")
    if any(path.exists() or path.is_symlink() for path in retired_paths):
        raise StateError("a task-owned directory marked for retirement still exists")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--revision", help="single proposed commit; defaults to main")
    parser.add_argument("--base", help="optional ancestor checkpoint; historical merges at or before it are preserved")
    parser.add_argument("--closeout", action="store_true", help="also inspect local working copy and residual state")
    parser.add_argument("--keep-workspace", action="append", default=[], help="explicitly retain an active secondary workspace")
    parser.add_argument("--retired-path", type=Path, action="append", default=[], help="task-owned directory that must be absent")
    parser.add_argument("--inherited-working-copy", help="exact pre-task working-copy commit to preserve")
    parser.add_argument("--keep-head", action="append", default=[], help="explicitly retain a pre-existing unreferenced head")
    parser.add_argument("--keep-worktree", type=Path, action="append", default=[], help="explicitly retain a pre-existing Git worktree")
    args = parser.parse_args(argv)
    if not args.closeout and (args.keep_workspace or args.retired_path or args.inherited_working_copy or args.keep_head or args.keep_worktree):
        parser.error("retention and retirement arguments require --closeout")
    try:
        root = args.repo_root.resolve()
        target, native = check_history(root, args.revision, args.base)
        if args.closeout:
            if not native:
                raise StateError("local closeout requires a jj workspace")
            check_closeout(root, target, {"default", *args.keep_workspace}, [p.absolute() for p in args.retired_path], args.inherited_working_copy, args.keep_head, args.keep_worktree)
    except StateError as error:
        print(f"VCS state check failed: {error}", file=sys.stderr)
        return 1
    retained = args.inherited_working_copy or args.keep_workspace or args.keep_head or args.keep_worktree
    closeout = ", local closeout with inherited work preserved" if retained else ", clean local closeout"
    print("VCS state check passed: linear history" + (closeout if args.closeout else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
