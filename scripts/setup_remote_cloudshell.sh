#!/usr/bin/env bash
set -euo pipefail

echo "=== [1/4] Installing system dependencies (tmate, rsync) ==="
sudo apt-get update -qq
sudo apt-get install -y -qq tmate rsync curl git

echo "=== [2/4] Registering Termux SSH Public Key ==="
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
TERMUX_PUBKEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG6jLWV93pZAeQuSQ/8VfA0MK3NfWKl5GRzUk7WAwkqE pankaj-termux"
if ! grep -q "$TERMUX_PUBKEY" "$HOME/.ssh/authorized_keys" 2>/dev/null; then
    echo "$TERMUX_PUBKEY" >> "$HOME/.ssh/authorized_keys"
    chmod 600 "$HOME/.ssh/authorized_keys"
    echo "Termux public key added to authorized_keys."
else
    echo "Termux public key already present."
fi

echo "=== [3/4] Installing Astral uv & Python ML environment in persistent HOME ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$HOME/workspace/fine_tuning"
cd "$HOME/workspace/fine_tuning"

# Initialize venv in persistent home if not present
if [ ! -d ".venv" ]; then
    uv venv .venv
fi

source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers peft datasets pyright

echo "=== [4/4] Starting tmate SSH Session ==="
echo "Copy the SSH command printed below into Termux:"
tmate -F
