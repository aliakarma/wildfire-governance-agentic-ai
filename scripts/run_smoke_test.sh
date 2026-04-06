#!/usr/bin/env bash
# =============================================================================
# Quick smoke test — verifies the installation is working.
# Must complete in under 60 seconds.
#
# Usage:
#   Bash:       bash scripts/run_smoke_test.sh
#   PowerShell: python -m pytest tests/smoke/ -v --timeout=60
# =============================================================================

set -euo pipefail

echo "=== Smoke Test ==="
echo "Expected: < 60 seconds | 7 tests"
echo ""

# Set PYTHONPATH if not already set
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src:."

if [[ -x ".venv-smoke/Scripts/python.exe" ]]; then
	PYTHON_BIN="./.venv-smoke/Scripts/python.exe"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
	PYTHON_BIN="./.venv/Scripts/python.exe"
else
	PYTHON_BIN="python"
fi

"$PYTHON_BIN" -m pytest tests/smoke/ -v --timeout=60

echo ""
echo "Smoke test passed! The installation is working correctly."
echo "To reproduce all paper results: make reproduce  (or: bash experiments/run_all.sh)"
