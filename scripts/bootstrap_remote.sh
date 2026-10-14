#!/usr/bin/env bash
set -euo pipefail

echo "=== [1/4] Installing Neovim & Base Tools ==="
apt-get update -qq
apt-get install -y -qq neovim curl git

echo "=== [2/4] Installing Python ML Libraries with uv ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv pip install --system --break-system-packages \
    torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --system --break-system-packages \
    transformers peft datasets pyright

echo "=== [3/4] Testing Imports ==="
python3 -c "import torch, transformers, peft; print('PEFT INSTALLED SUCCESSFULLY:', peft.__version__)"

echo "=== [4/4] Environment Ready ==="
