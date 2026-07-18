"""FastAPI application for the wildfire governance dashboard.

A thin adapter over the real ``wildfire_governance`` simulation. Run with:
    uvicorn dashboard.backend.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .benchmark import run_benchmark  # noqa: E402
from .render import episode_to_gif  # noqa: E402
from .schema import (  # noqa: E402
    METHOD_COLORS,
    METHOD_PRESETS,
    PARAM_SCHEMA,
    validate_and_default,
)
from .simulation_service import stream_episode  # noqa: E402

app = FastAPI(title="Wildfire Governance Dashboard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_PAPER_DIR = _REPO_ROOT / "results" / "paper"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class BenchmarkRequest(BaseModel):
    methods: List[str] = ["greedy_gomdp", "ppo_cmdp", "adaptive_ai", "static"]
    n_seeds: int = 5
    n_uavs: int = 16
    grid_size: int = 60
    n_timesteps: int = 300
    tau: float = 0.72


class ExportRequest(BaseModel):
    params: Dict[str, Any] = {}
    fps: int = 12
    theme: str = "dark"


# --------------------------------------------------------------------------- #
# REST
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "wildfire-governance-dashboard"}


@app.get("/api/config/schema")
def config_schema() -> Dict[str, Any]:
    return {"params": PARAM_SCHEMA, "methods": METHOD_PRESETS, "colors": METHOD_COLORS}


@app.get("/api/methods")
def methods() -> Dict[str, Any]:
    return {"methods": [
        {"id": mid, **meta, "color": METHOD_COLORS.get(mid, {})}
        for mid, meta in METHOD_PRESETS.items()
    ]}


@app.post("/api/benchmark")
async def benchmark(req: BenchmarkRequest) -> Dict[str, Any]:
    return await run_in_threadpool(
        run_benchmark, req.methods, req.n_seeds, req.n_uavs, req.grid_size, req.n_timesteps, req.tau
    )


@app.post("/api/export/gif")
async def export_gif(req: ExportRequest) -> Response:
    data = await run_in_threadpool(episode_to_gif, req.params, req.fps, "inferno", req.theme)
    return Response(content=data, media_type="image/gif",
                    headers={"Content-Disposition": "attachment; filename=episode.gif"})


@app.get("/api/breach-probability")
def breach_probability(n_validators: int = 7, max_byzantine: int = 2) -> Dict[str, Any]:
    """Theorem 2 breach probabilities: GOMDP (binomial tail) vs centralized.

    Reuses src/wildfire_governance/gomdp/breach_probability.py — real code, not
    hardcoded. When f exceeds the PBFT threshold floor((k-1)/3), consensus is not
    guaranteed and the GOMDP breach probability is reported as 1.0.
    """
    from wildfire_governance.gomdp.breach_probability import (
        compute_breach_probability_centralized,
        compute_breach_probability_gomdp,
    )

    threshold = (n_validators - 1) // 3
    bft_safe = max_byzantine <= threshold
    points = []
    for i in range(1, 11):
        p_c = round(i * 0.05, 2)
        gomdp = (
            compute_breach_probability_gomdp(n_validators, max_byzantine, p_c)
            if bft_safe
            else 1.0
        )
        points.append({
            "p_c": p_c,
            "gomdp": round(float(gomdp), 4),
            "central": round(float(compute_breach_probability_centralized(p_c)), 4),
        })
    return {
        "n_validators": n_validators,
        "max_byzantine": max_byzantine,
        "threshold": threshold,
        "bft_safe": bft_safe,
        "points": points,
        "note": "Theorem 2: P_breach(GOMDP) = P(>f of k validators compromised); "
                "centralized breach = per-channel attack probability.",
    }


@app.get("/api/paper-results/{table}")
def paper_results(table: str) -> Dict[str, Any]:
    """Serve a committed paper CSV, EXPLICITLY labeled as a reference value.

    Never blend this with live results (see Dashboard_Guide.md §19).
    """
    import csv
    safe = table.replace("/", "").replace("..", "")
    path = _PAPER_DIR / f"{safe}.csv"
    if not path.exists():
        return {"error": "not_found", "table": safe,
                "available": [p.stem for p in _PAPER_DIR.glob("*.csv")]}
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {"source": "paper_reference",
            "label": "Reference values from the manuscript (not a live run).",
            "table": safe, "rows": rows}


# --------------------------------------------------------------------------- #
# WebSocket — live episode stream
# --------------------------------------------------------------------------- #
@app.websocket("/ws/simulate")
async def ws_simulate(ws: WebSocket) -> None:
    await ws.accept()
    try:
        msg = await ws.receive_json()
        params = validate_and_default(msg.get("params"))
        await ws.send_json({"type": "start", "params": params})

        gen = stream_episode(params)
        # Pull CPU-bound frames off the event loop.
        while True:
            frame = await run_in_threadpool(next, gen, None)
            if frame is None:
                break
            await ws.send_json(frame)
            if frame.get("type") == "done":
                break
    except WebSocketDisconnect:
        return
    except Exception as exc:  # surface errors to the client instead of a silent drop
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Static frontend (production): serve the exported Next.js build if present.
# --------------------------------------------------------------------------- #
_FRONTEND_OUT = _REPO_ROOT / "dashboard" / "frontend" / "out"
if _FRONTEND_OUT.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_OUT), html=True), name="frontend")
