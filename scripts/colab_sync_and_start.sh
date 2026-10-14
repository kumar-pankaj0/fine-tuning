#!/usr/bin/env bash
set -euo pipefail

SCRIPTS_DIR="/data/data/com.termux/files/home/project/fine_tuning/scripts"
PROJECT_DIR="/data/data/com.termux/files/home/project/fine_tuning"
TMP_DIR="$PROJECT_DIR/scripts"

echo "=== [1/4] Checking / Provisioning Colab Server ==="
if ! colab server ls 2>/dev/null | grep -q "standard"; then
    echo "No active server found. Assigning 'Colab CPU'..."
    colab server assign
else
    echo "Active Colab server detected."
fi

echo "=== [2/4] Packaging & Syncing Neovim and Source Code ==="
# Package Neovim config
tar -czf "$TMP_DIR/nvim_config.tar.gz" -C "$HOME/.config" nvim
colab file upload "$TMP_DIR/nvim_config.tar.gz" /tmp/nvim_config.tar.gz
colab server run tar -xzf /tmp/nvim_config.tar.gz -C /root/.config
rm -f "$TMP_DIR/nvim_config.tar.gz"

# Package Project Source (excluding heavy output checkpoints)
tar -czf "$TMP_DIR/src_code.tar.gz" --exclude="output" --exclude="__pycache__" --exclude="*.bin" --exclude="*.safetensors" -C "$PROJECT_DIR" src
colab file upload "$TMP_DIR/src_code.tar.gz" /tmp/src_code.tar.gz
colab server run tar -xzf /tmp/src_code.tar.gz -C /root
rm -f "$TMP_DIR/src_code.tar.gz"

echo "=== [3/4] Ensuring Remote ML Stack & Pyright Are Installed ==="
colab file upload "$SCRIPTS_DIR/bootstrap_remote.sh" /tmp/bootstrap_remote.sh
colab server run bash /tmp/bootstrap_remote.sh

echo "=== [4/4] Launching Interactive Colab Shell ==="
echo "Entering remote shell. Run: cd /root/src && nvim fine_tuning/lora_runtime.py"
exec colab server shell
