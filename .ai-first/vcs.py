#!/usr/bin/env python3
"""main 이력과 명시적으로 종료하는 로컬 작업의 잔여물을 검사한다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
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
        raise StateError("full history is required for the local integration check")
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
            raise StateError("full history is required for the local integration check")
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


def workspace_inventory(root: Path) -> dict[str, dict[str, str]]:
    rows = jj(root, "workspace", "list", "-T", r'name ++ "\0" ++ root ++ "\0" ++ target.commit_id() ++ "\n"')
    current = revision_id(root, "@", True)
    records = [row.split("\0") for row in rows.splitlines()]
    result = {}
    for name, path, commit in records:
        if not path and commit == current and sum(row[2] == current for row in records) == 1:
            path = jj(root, "root")
        if not path or not Path(path).is_dir():
            raise StateError("workspace root is missing; classify existing work before starting")
        result[name] = {"path": str(Path(path).resolve()), "commit": commit}
    return result


def refs(root: Path, kind: str) -> dict[str, str]:
    records = jj(root, kind, "list", "-T", r'name ++ "\0" ++ normal_target.commit_id() ++ "\n"')
    result = dict(row.split("\0", 1) for row in records.splitlines())
    if any(not re.fullmatch(r"[0-9a-f]{40}", commit) for commit in result.values()):
        raise StateError("conflicted references must be resolved before starting")
    return result


def git_worktrees(root: Path) -> list[str]:
    if not (root / ".git").exists():
        return []
    return [str(Path(field[9:]).resolve()) for field in query(root, "git", "worktree", "list", "--porcelain", "-z").split("\0") if field.startswith("worktree ")]


def external_state_path(root: Path, path: Path) -> Path:
    path = path.absolute()
    if path.is_symlink() or path.resolve().is_relative_to(root):
        raise StateError("session state must be outside the repository and must not be a symlink")
    if any((parent / ".jj").exists() or (parent / ".git").exists() for parent in path.resolve().parents):
        raise StateError("session state must be outside all repository checkouts")
    return path


def write_state(path: Path, state: dict, *, create: bool = False) -> None:
    # 로컬 경로와 보존 ID는 private 파일로만 남긴다. 시작 기록은 덮어쓰지 않는다.
    if create:
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "w") as output:
            json.dump(state, output, indent=2)
            output.write("\n")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(state, output, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def start_session(root: Path, path: Path) -> None:
    if Path(query(root, "jj", "root")).resolve() != root:
        raise StateError("start from the canonical workspace root")
    query(root, "jj", "status")
    bookmark, base = load_policy(root)
    target, _ = check_history(root, bookmark)
    workspaces = workspace_inventory(root)
    for row in workspaces.values():
        query(Path(row["path"]), "jj", "status")
    workspaces = workspace_inventory(root)
    current = revision_id(root, "@", True)
    state = {
        "schema_version": 1, "root": str(root), "bookmark": bookmark, "base": base,
        "main": target, "working_copy": current,
        "inherited": jj(root, "log", "-r", "@", "--no-graph", "-T", "empty") != "true" or jj(root, "log", "-r", "parents(@)", "--no-graph", "-T", "commit_id") != target,
        "workspaces": workspaces, "worktrees": git_worktrees(root),
        "heads": jj(root, "log", "-r", "heads(all()) ~ (bookmarks() | tags() | working_copies())", "--no-graph", "-T", 'commit_id ++ "\n"').splitlines(),
        "tags": refs(root, "tag"), "bookmarks": refs(root, "bookmark"), "owned_paths": [],
    }
    write_state(path, state, create=True)


def load_session(root: Path, path: Path) -> dict:
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != 1 or state.get("root") != str(root):
        raise StateError("session belongs to a different repository or schema")
    if state.get("finished"):
        raise StateError("session is already finished; start a new task")
    if load_policy(root) != (state["bookmark"], state["base"]):
        raise StateError("VCS policy changed during the task; do not move the history checkpoint")
    return state


def own_copy(root: Path, state: dict, path: Path) -> None:
    path = path.absolute()
    protected = [root, *[Path(row["path"]) for row in state["workspaces"].values()], *map(Path, state["worktrees"])]
    if path.exists() or path.is_symlink() or any(path.resolve().is_relative_to(p) or p.is_relative_to(path.resolve()) for p in protected):
        raise StateError("register only a new external task copy before creating it")
    if str(path) in state["owned_paths"]:
        raise StateError("task copy is already registered")
    state["owned_paths"].append(str(path))


def is_ancestor(root: Path, ancestor: str, target: str) -> bool:
    return bool(jj(root, "log", "-r", f"{ancestor} & ancestors({target})", "--no-graph", "-T", "commit_id"))


def integrate_session(root: Path, state: dict, revision: str, remote: str | None) -> None:
    query(root, "jj", "status")
    target, _ = check_history(root, revision)
    current = revision_id(root, state["bookmark"], True)
    if not is_ancestor(root, state["main"], current) or not is_ancestor(root, current, target):
        raise StateError("candidate must extend current main; reapply only unpublished work linearly")
    if jj(root, "log", "-r", f"{current}..{target} & conflicts()", "--no-graph", "-T", "commit_id"):
        raise StateError("candidate contains unresolved conflicts")
    if remote:
        remotes = dict(line.split(maxsplit=1) for line in jj(root, "git", "remote", "list").splitlines())
        if remote not in remotes:
            raise StateError("unknown remote")
        record = query(root, "git", "ls-remote", "--exit-code", "--heads", remotes[remote], "refs/heads/" + state["bookmark"])
        remote_id = record.split()[0]
        cached = revision_id(root, state["bookmark"] + "@" + remote, True)
        if cached != remote_id or not is_ancestor(root, remote_id, target):
            raise StateError("remote advanced; fetch and revalidate unpublished work before publication")
    state["candidate"] = target


def finish_session(root: Path, state: dict) -> None:
    target = state.get("candidate")
    if not target:
        raise StateError("validate the completed candidate with integrate before finishing")
    check_history(root, target)
    current_refs = refs(root, "bookmark")
    current_tags = refs(root, "tag")
    if any(current_tags.get(name) != commit for name, commit in state["tags"].items()):
        raise StateError("pre-existing release tags changed")
    if any(current_refs.get(name) != commit for name, commit in state["bookmarks"].items() if name != state["bookmark"]):
        raise StateError("pre-existing bookmarked history changed")
    current_main = revision_id(root, state["bookmark"], True)
    check_history(root, current_main)
    if not is_ancestor(root, state["main"], current_main):
        raise StateError("pre-existing main history was rewritten")
    if not any(is_ancestor(root, target, commit) for commit in [*current_refs.values(), *current_tags.values()]):
        raise StateError("completed candidate needs a durable bookmark before retiring its copy")
    workspaces = workspace_inventory(root)
    for name, row in state["workspaces"].items():
        if name not in workspaces or workspaces[name]["path"] != row["path"]:
            raise StateError("a pre-existing workspace was removed or relocated")
        if row["path"] != str(root):
            query(Path(row["path"]), "jj", "status")
            if revision_id(Path(row["path"]), "@", True) != row["commit"]:
                raise StateError("pre-existing secondary work changed; preserve and classify concurrent work")
    if not set(state["worktrees"]).issubset(git_worktrees(root)):
        raise StateError("a pre-existing Git worktree registration was removed")
    check_closeout(root, target, set(state["workspaces"]), [Path(p) for p in state["owned_paths"]], state["working_copy"] if state["inherited"] else None, state["heads"], [Path(p) for p in state["worktrees"]])
    state["finished"] = True


def session_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Local agent task lifecycle; no automatic deletion or publication")
    parser.add_argument("action", choices=("start", "resume", "own", "integrate", "finish"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--state", type=Path, default=os.environ.get("AI_FIRST_VCS_SESSION"))
    parser.add_argument("--path", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--remote")
    args = parser.parse_args(argv)
    if not args.state:
        parser.error("--state or AI_FIRST_VCS_SESSION is required")
    if (args.action == "own") != bool(args.path) or (args.action == "integrate") != bool(args.revision) or (args.remote and args.action != "integrate"):
        parser.error("own requires --path; integrate requires --revision and accepts --remote")
    try:
        root = args.repo_root.resolve()
        path = external_state_path(root, args.state)
        if args.action == "start":
            start_session(root, path)
        else:
            # native helper가 보조 workspace에서 실행되어도 시작 기록의 canonical root를 쓴다.
            saved_root = Path(json.loads(path.read_text(encoding="utf-8"))["root"]).resolve()
            if saved_root != root:
                if args.action not in {"integrate", "resume"} or str(root) not in {row["path"] for row in workspace_inventory(saved_root).values()}:
                    raise StateError("run this action from the session's canonical workspace")
                if args.action == "integrate":
                    args.revision = revision_id(root, args.revision, True)
                root = saved_root
                external_state_path(root, path)
            state = load_session(root, path)
            if args.action == "own":
                own_copy(root, state, args.path)
            elif args.action == "integrate":
                integrate_session(root, state, args.revision, args.remote)
            elif args.action == "finish":
                finish_session(root, state)
            write_state(path, state)
    except (StateError, OSError, ValueError, KeyError, TypeError) as error:
        # private manifest의 원문이나 경로를 오류 메시지로 노출하지 않는다.
        message = str(error) if isinstance(error, StateError) else "cannot read or update the private session state"
        print(f"VCS task {args.action} failed: {message}", file=sys.stderr)
        return 1
    print(f"VCS task {args.action} passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"start", "resume", "own", "integrate", "finish"}:
        return session_main(argv)
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
