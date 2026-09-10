#!/bin/sh
set -eu

repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

fail() {
  printf 'agent harness interface check failed: %s\n' "$*" >&2
  exit 1
}

for required_file in \
  .ai-first.toml \
  .ai-first.lock \
  .ai-first/check.py \
  .ai-first/overlays/agents-first-read.md \
  .ai-first/overlays/agents-project.md \
  .ai-first/overlays/harness-project.md \
  AGENTS.md \
  docs/agent-harness.md \
  docs/HANDOFF.md \
  scripts/check-publication-boundary.py; do
  [ -s "$required_file" ] || fail "missing or empty $required_file"
done

# 공통 generated interface와 선언/lock 정합성은 pinned standalone checker에 위임한다.

grep -Fq -- '"source_kind": "release"' .ai-first.lock ||
  fail "framework release source is missing from lock"

python3 .ai-first/check.py
scripts/check-publication-boundary.py --self-test
scripts/check-publication-boundary.py

printf 'agent harness interface is valid: ai-first-harness-v1 / go-mfcc overlay\n'
