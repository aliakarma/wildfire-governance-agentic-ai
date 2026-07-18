"""Dashboard backend package.

A thin FastAPI adapter over the existing ``wildfire_governance`` package.
It never fabricates numbers: every metric streamed to the UI is computed by
the real simulation (mirroring ``experiments/utils/runner.py``).
"""
