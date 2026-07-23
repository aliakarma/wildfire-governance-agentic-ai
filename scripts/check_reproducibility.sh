#!/usr/bin/env bash
# =============================================================================
# Check that newly computed results match pre-committed paper values
# within 5% relative tolerance.
#
# Usage:
#   bash scripts/check_reproducibility.sh
# =============================================================================

set -euo pipefail

if [ -z "${PYTHON_BIN:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "[FAIL] Python interpreter not found in PATH (tried python3, python)."
        exit 1
    fi
fi

exec "$PYTHON_BIN" scripts/check_reproducibility.py "$@"
