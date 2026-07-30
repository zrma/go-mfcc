#!/bin/sh
set -eu

repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

scripts/check-agent-harness-interface.sh

unformatted=$(
  find mfcc -type f -name '*.go' -print0 |
    xargs -0 gofmt -l
)
if [ -n "$unformatted" ]; then
  printf 'gofmt required:\n%s\n' "$unformatted" >&2
  exit 1
fi

go vet ./...
go test ./...

printf 'go-mfcc checks passed\n'
