#!/bin/bash
# Setup script for TESLearn using uv

set -e

echo "Setting up TESLearn environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "✓ uv is installed"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

echo "✓ Virtual environment ready"

# Activate and install
source .venv/bin/activate

echo "Installing dependencies..."
uv pip install -e

echo "✓ Dependencies installed"

# Setup Jupyter kernel
python -m ipykernel install --user --name teslearn --display-name "TESLearn"

echo "✓ Jupyter kernel registered"

echo ""
echo "Setup complete! Activate the environment with:"
echo "   source .venv/bin/activate"
echo ""
echo "To run the tutorial:"
echo "   jupyter notebook tutorial.ipynb"
