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

@app.on_event("startup")
def startup_event():
    print("\n" + "="*80)
    print("  SECURITY WARNING: Wildfire Governance Dashboard Backend is running.")
    print("  This API has NO AUTHENTICATION and should only be run on localhost.")
    print("  DO NOT EXPOSE THIS SERVICE TO THE PUBLIC INTERNET.")
    print("="*80 + "\n")

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


# Canonical catalog of every manuscript artifact — the single list the frontend
# uses to surface all experiments. Mirrors results/paper/MANIFEST.yaml,
# PROVENANCE.md and scripts/check_reproducibility.py.
#
# provenance classes (identical to PROVENANCE.md):
#   exact          closed-form / deterministic; reproduces bit-for-bit
#   measured       aggregated over seeds 0-19 by the simulation core
#   specification  a configuration table, not a measurement
#   training-derived  from PPO training runs, not the evaluation aggregator
#   supplementary  NOT in the manuscript; badged accordingly
#
# Artifacts withdrawn from the manuscript (SafeDreamer/CCPO, the CNN ablation,
# the Fabric microbenchmark) are absent by design — see the `withdrawn` block of
# results/paper/MANIFEST.yaml.
_ARTIFACT_CATALOG: List[Dict[str, Any]] = [
    {"id": "table1_rl_comparison", "title": "Policy comparison (9 configurations)", "kind": "table", "paper_ref": "Table 1", "route": "Benchmark", "provenance": "measured", "metrics": ["ld_mean", "fp_mean", "fn_mean", "compliance_pct"]},
    {"id": "table1_rl_comparison_main", "title": "Full-metric comparison", "kind": "table", "paper_ref": "Table 5 (appendix)", "route": "Benchmark", "provenance": "measured", "metrics": ["ld_mean", "fp_mean", "bc_delay_mean", "gov_overhead_pct"]},
    {"id": "table2_ablation", "title": "Component ablation", "kind": "table", "paper_ref": "Table 2", "route": "Ablation", "provenance": "measured", "metrics": ["ld_mean", "fp_mean", "injections_blocked"]},
    {"id": "table3_adversarial", "title": "Adversarial robustness", "kind": "table", "paper_ref": "Table 6 (appendix)", "route": "Adversarial", "provenance": "measured", "metrics": ["gomdp", "central_sig", "central"]},
    {"id": "table4_realworld_viirs", "title": "VIIRS-data evaluation (3 events)", "kind": "table", "paper_ref": "Table 7 (appendix)", "route": "VIIRS", "provenance": "measured", "metrics": ["ld_mean", "fp_mean", "gov_compliance_pct"]},
    {"id": "table5_byzantine_theory", "title": "Validator compromise — stochastic theory", "kind": "table", "paper_ref": "Table 8 (appendix)", "route": "Adversarial", "provenance": "exact", "metrics": ["p_break_gomdp", "p_break_sig"]},
    {"id": "table5_byzantine_empirical", "title": "Validator compromise — deterministic", "kind": "table", "paper_ref": "Table 8 (appendix)", "route": "Adversarial", "provenance": "exact", "metrics": ["breach", "fp_pct"]},
    {"id": "table6_ksweep", "title": "Validator-count sweep (p_c = 0.10)", "kind": "table", "paper_ref": "Table 9 (appendix)", "route": "Adversarial", "provenance": "exact", "metrics": ["p_break_gomdp_theory", "empirical"]},
    {"id": "table7_hitl_sensitivity", "title": "HITL operator-error sensitivity", "kind": "table", "paper_ref": "Table 10 (appendix)", "route": "HITL", "provenance": "measured", "metrics": ["fn_mean", "fp_mean", "gov_compliance_pct"]},
    {"id": "table9_multisig", "title": "m-of-n multisignature ablation", "kind": "table", "paper_ref": "Table 11 (appendix)", "route": "Adversarial", "provenance": "exact", "metrics": ["ld_mean", "fp_mean", "injections_blocked"]},
    {"id": "statistical_tests", "title": "Significance and equivalence tests", "kind": "table", "paper_ref": "Sec. 6.2 (statistical testing)", "route": "Benchmark", "provenance": "exact", "metrics": ["statistic", "p_value", "effect_size_d"]},
    {"id": "table_config_parameters", "title": "Experimental configuration parameters", "kind": "table", "paper_ref": "Table 4 (appendix)", "route": "Experiments", "provenance": "specification", "metrics": ["value"]},
    {"id": "fig2_false_alerts", "title": "False-alert rate vs fleet size", "kind": "figure", "paper_ref": "Figure 2", "route": "Scalability", "provenance": "measured", "metrics": ["fp_mean"]},
    {"id": "fig3_learning_curve", "title": "PPO-GOMDP training convergence", "kind": "figure", "paper_ref": "Sec. 5.2 (training)", "route": "Learning", "provenance": "training-derived", "metrics": ["ld_mean", "greedy_baseline"]},
    {"id": "fig3_latency_data", "title": "Detection latency vs fleet size", "kind": "figure", "paper_ref": "Figure 3", "route": "Scalability", "provenance": "measured", "metrics": ["ld_mean", "ld_std"]},
    {"id": "figure3_tradeoff_frontier", "title": "Latency / false-alert tradeoff at N=40", "kind": "figure", "paper_ref": "SUPPLEMENTARY (not in the paper)", "route": "Benchmark", "provenance": "supplementary", "metrics": ["ld_mean", "fp_mean"]},
    {"id": "fig5_tradeoff_data", "title": "Tradeoff source data at N=40", "kind": "figure", "paper_ref": "SUPPLEMENTARY (not in the paper)", "route": "Benchmark", "provenance": "supplementary", "metrics": ["ld_mean", "fp_mean"]},
]


@app.get("/api/artifacts")
def artifacts() -> Dict[str, Any]:
    """Enumerate every manuscript table/figure the dashboard can surface, with a
    committed-CSV present check and provenance tag. The frontend uses this to
    render a view for every experiment (see /api/paper-results/{id} for data)."""
    out = []
    for a in _ARTIFACT_CATALOG:
        csv_present = (_PAPER_DIR / f"{a['id']}.csv").exists()
        out.append({**a, "csv_present": csv_present})
    return {"artifacts": out,
            "note": "Values are the manuscript's, verified cell-by-cell against "
                    "Paper/AAAI/Wildfire.tex by scripts/verify_paper_alignment.py. "
                    "provenance: exact=closed-form/deterministic; "
                    "measured=aggregated over seeds 0-19; "
                    "specification=configuration, not a measurement; "
                    "training-derived=from PPO training runs; "
                    "supplementary=NOT in the manuscript."}


_ARTIFACT_BY_ID = {a["id"]: a for a in _ARTIFACT_CATALOG}


@app.get("/api/paper-results/{table}")
def paper_results(table: str) -> Dict[str, Any]:
    """Serve a committed manuscript CSV, EXPLICITLY labeled as a paper value.

    Never blend this with live results (see Dashboard_Guide.md §19).
    """
    import csv
    safe = table.replace("/", "").replace("..", "")
    path = _PAPER_DIR / f"{safe}.csv"
    if not path.exists():
        return {"error": "not_found", "table": safe,
                "available": sorted(p.stem for p in _PAPER_DIR.glob("*.csv"))}
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    meta = _ARTIFACT_BY_ID.get(safe, {})
    return {"source": "paper_reference",
            "label": "Values as published in the manuscript (not a live run).",
            "table": safe,
            "paper_ref": meta.get("paper_ref"),
            "provenance": meta.get("provenance"),
            "rows": rows}


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
#
# The build lives in dashboard/frontend/out/, which is git-ignored — so a fresh
# clone has no UI. Rather than let the mount silently disappear and return a bare
# {"detail":"Not Found"} at "/", we detect a missing/partial build and serve a
# clear "how to build" page. This is the #1 "nothing works" trap for a reviewer
# who runs uvicorn without first building the frontend.
# --------------------------------------------------------------------------- #
_FRONTEND_OUT = _REPO_ROOT / "dashboard" / "frontend" / "out"
_FRONTEND_INDEX = _FRONTEND_OUT / "index.html"

_BUILD_HELP_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wildfire GOMDP Dashboard — frontend not built</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         background:#0B0E14; color:#E8EDF4; line-height:1.6; }
  main { max-width:760px; margin:0 auto; padding:56px 24px; }
  h1 { font-size:24px; margin:0 0 4px; }
  .tag { display:inline-block; font-size:12px; font-weight:600; color:#FFB067;
         border:1px solid #F2B455; border-radius:999px; padding:2px 10px; margin-bottom:20px; }
  p { color:#9AA6B8; }
  code, pre { font-family:IBM Plex Mono,ui-monospace,SFMono-Regular,Menlo,monospace; }
  pre { background:#12161F; border:1px solid #263041; border-radius:12px;
        padding:16px 18px; overflow-x:auto; color:#E8EDF4; font-size:13.5px; }
  .ok { color:#35D0A5; } .step { margin-top:28px; }
  a { color:#6FB1FF; }
</style></head><body><main>
  <div class="tag">FRONTEND NOT BUILT</div>
  <h1>🔥 Wildfire GOMDP Dashboard</h1>
  <p>The backend API is running, but the Next.js frontend has not been built yet, so
     there is no UI to serve. The build output (<code>dashboard/frontend/out/</code>)
     is git-ignored and is not present in this checkout.</p>

  <div class="step"><strong>Build the UI once (needs Node.js 18+):</strong></div>
  <pre>cd dashboard/frontend
npm install
npm run build
cd ../..</pre>

  <div class="step"><strong>Then restart this server and reload:</strong></div>
  <pre>python -m uvicorn dashboard.backend.main:app --host 127.0.0.1 --port 8123</pre>

  <p class="step">Or use the one-command launcher, which builds the UI automatically
     if it is missing:</p>
  <pre>python dashboard/run_dashboard.py --port 8123</pre>

  <p class="step ok">The API itself is healthy — try
     <a href="/api/health">/api/health</a> or <a href="/docs">/docs</a>.</p>
</main></body></html>"""


def _frontend_is_built() -> bool:
    return _FRONTEND_INDEX.is_file()


if _frontend_is_built():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_OUT), html=True), name="frontend")
else:
    from fastapi.responses import HTMLResponse

    print(
        "\n  NOTE: frontend build not found at "
        f"{_FRONTEND_OUT} — serving build instructions at '/'.\n"
        "        Build it with:  cd dashboard/frontend && npm install && npm run build\n"
        "        or run:          python dashboard/run_dashboard.py\n"
    )

    @app.get("/", response_class=HTMLResponse)
    def _frontend_not_built() -> HTMLResponse:
        return HTMLResponse(_BUILD_HELP_HTML, status_code=503)

    # Catch every other non-API path (e.g. a deep-linked screen) with the same
    # help page. Registered last, so the /api/* and /ws/* routes above win.
    @app.get("/{_path:path}", response_class=HTMLResponse)
    def _frontend_not_built_catchall(_path: str) -> HTMLResponse:
        return HTMLResponse(_BUILD_HELP_HTML, status_code=503)
