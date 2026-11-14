#!/bin/bash
# Run on your MAC (not on the cluster). Copies the code, and optionally the data zips, to Lawrence.
#   bash cluster/sync_to_cluster.sh code            # code only (seconds)
#   bash cluster/sync_to_cluster.sh tf3             # + ToothFairy3.zip (27 GB, hours on wifi; resumable)
#   bash cluster/sync_to_cluster.sh small           # + CONTACT zip (16 MB)
#   bash cluster/sync_to_cluster.sh dentex          # + Zenodo/DENTEX zips (10 GB)
# Uses one SSH master connection, so Duo is asked once per run.
set -euo pipefail
USER_AT_HOST="${USER_AT_HOST:-dipesh.tripathi@lawrence.usd.edu}"
REMOTE_PROJ="${REMOTE_PROJ:-/home/santosh_lab/shared/dipesh/dental}"   # only shared/ is group-writable
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
CTL="$HOME/.ssh/cm-lawrence-%r@%h:%p"
SSH="ssh -o ControlMaster=auto -o ControlPath=$CTL -o ControlPersist=2h"
# macOS ships openrsync (no --append-verify / --info). With GNU rsync (brew install rsync) big files resume
# where they stopped; with openrsync --partial keeps the partial file but the transfer restarts.
if rsync --version 2>/dev/null | grep -q openrsync; then
  RS=(rsync -a --partial --progress); RS_BIG=("${RS[@]}")
else
  RS=(rsync -a --partial --info=progress2); RS_BIG=("${RS[@]}" --append-verify)
fi

$SSH "$USER_AT_HOST" "mkdir -p '$REMOTE_PROJ/data_raw/Zenodo'"
# code (no data, envs, outputs, renders) + data/derived (small label tables, e.g. contact3d.json)
"${RS[@]}" -z -e "$SSH" \
  --include 'data/' --include 'data/derived/***' --exclude 'data/*' --exclude '.venv/' --exclude 'outputs/' --exclude 'external/' --exclude '*.zip' \
  --exclude 'Zenodo/' --exclude '__pycache__/' --exclude '.DS_Store' \
  "$LOCAL/" "$USER_AT_HOST:$REMOTE_PROJ/code/"

for what in "$@"; do
  case "$what" in
    code) ;;
    tf3)    "${RS_BIG[@]}" -e "$SSH" "$LOCAL/ToothFairy3.zip" "$USER_AT_HOST:$REMOTE_PROJ/data_raw/" ;;
    small)  "${RS[@]}" -e "$SSH" "$LOCAL/CONTACTdataset.zip" "$USER_AT_HOST:$REMOTE_PROJ/data_raw/" ;;
    dentex) "${RS_BIG[@]}" -e "$SSH" "$LOCAL/Zenodo/" "$USER_AT_HOST:$REMOTE_PROJ/data_raw/Zenodo/" ;;
    *) echo "unknown target $what"; exit 1 ;;
  esac
done
echo "done. On the cluster, data root = $REMOTE_PROJ/data_raw"
