#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="${1:-/c/Users/Master/Desktop/EPFL/red_queen_moderation}"
REMOTE_DIR="/home/mstojano/red_queen"
TMP_DIR="$(mktemp -d)"
PACKAGE_DIR="$TMP_DIR/red_queen"

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "Local directory does not exist: $LOCAL_DIR" >&2
  exit 1
fi

mkdir -p "$PACKAGE_DIR"
rsync -a \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'data' \
  --exclude 'outputs' \
  "$LOCAL_DIR"/ "$PACKAGE_DIR"/

ssh rcp "rm -rf '$REMOTE_DIR'"
scp -r "$PACKAGE_DIR" rcp:/home/mstojano/
rm -rf "$TMP_DIR"

echo "Copied $LOCAL_DIR to rcp:$REMOTE_DIR"
echo "Verify with: ssh rcp 'grep Baseline /home/mstojano/red_queen/src/baseline_evaluation.py | head'"
