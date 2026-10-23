#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./connect_cloudshell.sh "ABCD1234EFGH@ny1.tmate.io"
# Or:
#   ./connect_cloudshell.sh "ssh ABCD1234EFGH@ny1.tmate.io"

SSH_TARGET="${1:-}"

if [[ -z "$SSH_TARGET" ]]; then
    echo "Usage: $0 <tmate-ssh-target>"
    echo "Example: $0 ABCD1234EFGH@ny1.tmate.io"
    exit 1
fi

SSH_TARGET="${SSH_TARGET#ssh }"

echo "Connecting to Cloud Shell via SSH with ServerAliveInterval keepalive..."
exec ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=5 $SSH_TARGET
