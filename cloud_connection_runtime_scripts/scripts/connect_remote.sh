#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./connect_remote.sh <subdomain>.trycloudflare.com
# Or:
#   ./connect_remote.sh https://<subdomain>.trycloudflare.com

TARGET="${1:-}"

if [[ -z "$TARGET" ]]; then
    echo "Usage: $0 <trycloudflare-hostname>"
    echo "Example: $0 random-words-123.trycloudflare.com"
    exit 1
fi

# Strip https:// or http:// or trailing slashes
HOSTNAME=$(echo "$TARGET" | sed -e 's|^[^/]*//||' -e 's|/.*$||')

echo "Connecting to remote VM at $HOSTNAME via Cloudflare Tunnel..."
exec ssh -o ServerAliveInterval=30 \
         -o ServerAliveCountMax=5 \
         -o StrictHostKeyChecking=no \
         -o UserKnownHostsFile=/dev/null \
         -o ProxyCommand="cloudflared access ssh --hostname %h" \
         root@"$HOSTNAME"
