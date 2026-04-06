#!/usr/bin/env bash
# =============================================================================
# One-command environment setup for wildfire-governance-agentic-ai.
#
# Usage:
#   Bash (Linux/macOS):  bash scripts/setup_environment.sh
#   Git Bash (Windows):  bash scripts/setup_environment.sh
#
# For PowerShell (Windows), use the conda commands directly:
#   conda env create -f environment.yml
#   conda activate wildfire-gov
#   pip install -e ".[dev]"
# =============================================================================

set -euo pipefail

echo "=== Wildfire Governance — Environment Setup ==="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED="3.10"
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo "[OK] Python $PYTHON_VERSION >= $REQUIRED"
else
    echo "[ERROR] Python >= $REQUIRED required. Found: $PYTHON_VERSION"
    echo "Download at: https://www.python.org/downloads/"
    exit 1
fi

# Option A: conda (preferred)
if command -v conda &>/dev/null; then
    echo ""
    echo "--- Setting up conda environment ---"
    conda env create -f environment.yml --force
    echo ""
    echo "Activate with: conda activate wildfire-gov"
    echo "Then run:      pip install -e '[dev]'"
else
    # Option B: pip + venv
    echo ""
    echo "--- Setting up pip virtual environment (conda not found) ---"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements-dev.txt
    pip install -e ".[dev]"
    echo ""
    echo "Activate with: source .venv/bin/activate"
fi

echo ""
echo "[OK] Environment ready. Run smoke test:"
echo "     make test-smoke"
echo ""
echo "     OR:"
echo "     python -m pytest tests/smoke/ -v --timeout=60"
