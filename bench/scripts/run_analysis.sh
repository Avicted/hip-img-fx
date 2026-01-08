#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/hip_img_fx_venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
python -m pip install --upgrade pip --quiet
python -m pip install -r "$SCRIPT_DIR/requirements.txt" --quiet

echo ""
python "$SCRIPT_DIR/analyze_results.py" "$@"

deactivate
