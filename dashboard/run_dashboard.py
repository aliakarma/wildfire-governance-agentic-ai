#!/usr/bin/env python
"""One-command launcher for the Wildfire GOMDP dashboard.

Builds the Next.js frontend if it hasn't been built yet (the ``out/`` directory
is git-ignored, so a fresh clone has no UI), then starts the FastAPI server that
serves both the API and the static UI.

Usage:
    python dashboard/run_dashboard.py                 # port 8123
    python dashboard/run_dashboard.py --port 8000
    python dashboard/run_dashboard.py --skip-build    # assume UI already built
    python dashboard/run_dashboard.py --force-build   # rebuild even if present

This removes the most common "nothing works" trap: running the backend without
first building the frontend, which otherwise serves a bare 404 at '/'.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = REPO_ROOT / "dashboard" / "frontend"
FRONTEND_OUT = FRONTEND / "out"
FRONTEND_INDEX = FRONTEND_OUT / "index.html"


def _npm() -> str | None:
    """Locate the npm executable (npm.cmd on Windows)."""
    return shutil.which("npm") or shutil.which("npm.cmd")


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"  $ {' '.join(cmd)}   (in {cwd})")
    subprocess.run(cmd, cwd=str(cwd), check=True, shell=False)


def build_frontend() -> None:
    npm = _npm()
    if npm is None:
        sys.exit(
            "ERROR: Node.js/npm not found on PATH. Install Node.js 18+ and retry,\n"
            "       or build the UI manually and re-run with --skip-build."
        )
    if not (FRONTEND / "node_modules").exists():
        print("- Installing frontend dependencies (first run)...")
        _run([npm, "install"], cwd=FRONTEND)
    print("- Building the frontend (Next.js static export)...")
    _run([npm, "run", "build"], cwd=FRONTEND)
    if not FRONTEND_INDEX.is_file():
        sys.exit(f"ERROR: build finished but {FRONTEND_INDEX} is missing.")
    print("[ok] Frontend built.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Wildfire GOMDP dashboard.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--reload", action="store_true", help="uvicorn auto-reload (dev)")
    ap.add_argument("--skip-build", action="store_true", help="do not build the UI")
    ap.add_argument("--force-build", action="store_true", help="rebuild the UI")
    args = ap.parse_args()

    if args.force_build or (not args.skip_build and not FRONTEND_INDEX.is_file()):
        build_frontend()
    elif FRONTEND_INDEX.is_file():
        print("[ok] Using existing frontend build (dashboard/frontend/out/).")
    else:
        print("! --skip-build set but no build found; the server will show build help.")

    # Import-time side effects (sys.path setup) live in the app module; run via
    # the uvicorn CLI so the module import path matches the documented command.
    import uvicorn  # noqa: PLC0415

    print(f"\n-> Open http://{args.host}:{args.port}/\n")
    uvicorn.run(
        "dashboard.backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(REPO_ROOT),
    )


if __name__ == "__main__":
    main()
