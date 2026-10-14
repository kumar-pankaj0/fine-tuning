#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./colab_pull_file.sh fine_tuning/lora_runtime.py

FILE_REL="${1:-}"

if [[ -z "$FILE_REL" ]]; then
    echo "Usage: $0 <relative-path-inside-src>"
    echo "Example: $0 fine_tuning/lora_runtime.py"
    exit 1
fi

LOCAL_PATH="/data/data/com.termux/files/home/project/fine_tuning/src/$FILE_REL"
mkdir -p "$(dirname "$LOCAL_PATH")"

colab server run cat "/root/src/$FILE_REL" > "$LOCAL_PATH"
echo "Successfully pulled /root/src/$FILE_REL -> $LOCAL_PATH"
