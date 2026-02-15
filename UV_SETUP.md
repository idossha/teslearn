# UV Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package management.

## Quick Start

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or on macOS with Homebrew:
```bash
brew install uv
```

### 2. Create and activate the environment

```bash
# Create virtual environment with the pinned Python version (3.11)
uv venv

# Activate the environment
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
# Install the package with core dependencies
uv pip install -e .

# Install with optional dependencies
uv pip install -e ".[plotting,jupyter]"     # For visualization and notebooks
uv pip install -e ".[dev]"                  # For development (testing, linting)
uv pip install -e ".[all]"                  # Install everything
```

### 4. Run the tutorial notebook

```bash
# Install jupyter dependencies
uv pip install -e ".[jupyter]"

# Register the kernel
python -m ipykernel install --user --name teslearn --display-name "TESLearn"

# Launch jupyter
jupyter notebook tutorial.ipynb
```

## Common Commands

```bash
# Sync dependencies with lock file
uv pip sync requirements.txt

# Generate lock file
uv pip compile pyproject.toml -o requirements.txt

# Update all packages
uv pip install -e ".[all]" --upgrade

# Run tests
uv run pytest

# Run Python script
uv run python script.py
```

## Using without activating environment

You can run commands directly using `uv run`:

```bash
uv run python -c "from teslearn import Subject; print('Works!')"
uv run pytest
uv run jupyter notebook
```

## Environment Details

- **Python version**: 3.11 (see `.python-version`)
- **Package manager**: uv (replaces pip)
- **Virtual environment**: `.venv/` (auto-managed by uv)

## Troubleshooting

### Permission denied on macOS/Linux
```bash
chmod +x $(which uv)
```

### Reset environment
```bash
rm -rf .venv
uv venv
uv pip install -e ".[all]"
```

### Check Python path
```bash
which python  # Should point to .venv/bin/python
uv pip list   # Shows installed packages
```

## VS Code Integration

1. Install the Python extension
2. Press `Cmd/Ctrl + Shift + P` → "Python: Select Interpreter"
3. Choose the `.venv` environment

## Why uv?

- **10-100x faster** than pip
- **Built-in virtual environment management**
- **Universal lock files** for reproducible builds
- **Drop-in replacement** for pip and virtualenv

For more information, visit: https://docs.astral.sh/uv/