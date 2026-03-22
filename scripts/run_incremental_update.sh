#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
POOL_SIZE="${POOL_SIZE:-100}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$ROOT_DIR"
. .venv/bin/activate
PYTHONPATH=src python scripts/incremental_update_clickhouse.py --pool-size "$POOL_SIZE" $EXTRA_ARGS
