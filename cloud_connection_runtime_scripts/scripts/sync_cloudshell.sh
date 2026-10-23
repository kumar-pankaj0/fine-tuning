#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./sync_cloudshell.sh pull "ssh-target"
#   ./sync_cloudshell.sh push "ssh-target"
#
# Examples:
#   ./sync_cloudshell.sh pull user@ny1.tmate.io
#   ./sync_cloudshell.sh push "-p 2222 user@host"

DIRECTION="${1:-}"
SSH_TARGET="${2:-}"

if [[ -z "$DIRECTION" || -z "$SSH_TARGET" ]]; then
    echo "Usage: $0 [pull|push] <ssh-target-or-tmate-string>"
    echo "Example: $0 pull ABCD1234EFGH@ny1.tmate.io"
    exit 1
fi

# Clean up SSH target if passed as full "ssh ABCD@..." command
SSH_TARGET="${SSH_TARGET#ssh }"

LOCAL_DIR="/data/data/com.termux/files/home/project/fine_tuning/src/"
REMOTE_DIR="~/workspace/fine_tuning/src/"

# Exclude virtual environments and cache directories from syncing
EXCLUDES=(
    "--exclude=.venv"
    "--exclude=__pycache__"
    "--exclude=.git"
    "--exclude=.pytest_cache"
    "--exclude=*.pyc"
)

if [[ "$DIRECTION" == "pull" ]]; then
    echo "[SYNC] Pulling remote changes from Cloud Shell to local Termux..."
    rsync -avz --progress "${EXCLUDES[@]}" -e ssh "${SSH_TARGET}:${REMOTE_DIR}" "${LOCAL_DIR}"
    echo "[SYNC] Pull complete. Local files updated."
elif [[ "$DIRECTION" == "push" ]]; then
    echo "[SYNC] Pushing local changes from Termux to Cloud Shell..."
    rsync -avz --progress "${EXCLUDES[@]}" -e ssh "${LOCAL_DIR}" "${SSH_TARGET}:${REMOTE_DIR}"
    echo "[SYNC] Push complete. Remote files updated."
else
    echo "Error: Unknown direction '$DIRECTION'. Use 'pull' or 'push'."
    exit 1
fi
