# Wildfire Governance Dashboard

Interactive dashboard for the GOMDP wildfire-monitoring project. It drives the
**real** `wildfire_governance` simulation (not the pre-committed result CSVs) and
streams every step to an animated, bilingual, themeable UI.

Built per [`../Dashboard_Guide.md`](../Dashboard_Guide.md). This covers **Phase 0–3**
plus part of Phase 4 (Live Simulation, Governance Explorer, Adversarial Lab,
Benchmarks/Reproducibility, **A/B Compare**, and **shareable permalinks**). See
[Status](#status) for what's implemented vs. planned.

---

## What works today

- **Live episode streaming** — a FastAPI backend runs the real simulation one
  timestep at a time (mirroring `experiments/utils/runner.py`) and streams frames
  over WebSocket. Nothing is hardcoded; the numbers on screen are computed live.
- **Animated grid canvas** — inferno heat map + ground-truth fire outline + UAV
  markers (with battery-coloured rings) + pulsing alert/injection markers.
- **Real metric HUD** — detection latency `L_d`, false-alert rate `F_p`,
  governance compliance %, injections blocked — updated every frame.
- **Interactive parameters** — method preset, grid size, UAV fleet, sectors,
  timesteps, confidence threshold τ, seed, and adversarial controls (spoofing,
  injection, Byzantine faults, packet drop, sensor failure).
- **Playback** — play/pause/step, scrubber, 0.5×–8× speed.
- **Governance ledger** — streaming APPROVED / BLOCKED / HITL-rejected /
  injection-blocked events with certificate hashes.
- **Governance Explorer** — live predicate inspector rendering
  `G = [Conf > τ] ∧ HA ∧ sig_valid ∧ consensus` with each term evaluated
  green/red against the real contract result, a PBFT validator ring (Byzantine
  nodes highlighted, BFT-safety indicator), and a clickable audit log.
- **Adversarial Lab** — attack presets (spoofing / injection / Byzantine),
  a safety verdict (injections attempted vs blocked, invariant held/breached),
  and a Theorem-2 breach-probability chart (GOMDP binomial tail vs centralized),
  computed by the real `breach_probability` module.
- **Benchmark Explorer** — pick methods, then **Live compute** (runs seeds now
  with mean ± 95% CI) or **Paper reference** (committed manuscript values,
  clearly labeled). Grouped L_d / F_p / compliance charts plus the Fig-5
  latency–false-alert tradeoff frontier.
- **Reproducibility diff** — an honest live-vs-paper table with 5%-tolerance
  shading and raw per-seed CSV download. It surfaces real deviations (e.g. the
  live simulation's L_d/F_p do not match the manuscript's hardcoded values)
  rather than hiding them — see [Scientific integrity](#scientific-integrity).
- **PyroRL-style grid** — green grass on a visible white gridline, fire drawn
  as a warm yellow→orange→red gradient keyed by how long each cell has burned.
  The live viewer uses a deliberately gentle fire model (P_spread ~0.03–0.05 at
  mean field conditions vs the paper model's ~0.55) so a 500-step episode burns
  ~15% of the grid as a connected, growing front the fleet can search, verify and
  encircle — with per-UAV flight-path trails, heading cues and battery rings.
- **VIIRS real-world screen** — three VIIRS-observed events
  (California ’20 / Mediterranean ’21 / NSW ’19–20) with a live regional
  simulation and the manuscript's VIIRS-validation reference metrics (labeled).
- **A paper-coverage view for every table & figure** — beyond the core screens,
  dedicated views surface each manuscript artifact with a provenance badge
  (Exact / Calibration / Reference / Supplementary):
  - **All Experiments** — every one of the paper's artifacts as provenance-badged
    cards, sourced live from `/api/artifacts`.
  - **Ablation** (Table 2) — component-knockout L_d/F_p bars + injections-blocked.
  - **Scalability** (Fig 2 & 4) — F_p-vs-N and L_d-vs-N with the Proposition-1 bound.
  - **Learning** (Fig 3) — PPO-GOMDP validation-L_d curve vs the greedy baseline.
  - **HITL** (Table 7) — FN_r/F_p vs operator error p_err, compliance pinned 100%.
  - **CNN** (Table 10) — MLP-vs-CNN architecture comparison (labeled *reference*).
  - **Adversarial → Consensus reference** — closed-form byzantine + k-sweep +
    multisig (Tables for validator compromise / k-sweep / multisig), badged *Exact*.
  The tradeoff-frontier and stress views carry a *Supplementary — not in the paper*
  badge. Map: [../PROVENANCE.md](../PROVENANCE.md) · [../results/paper/MANIFEST.yaml](../results/paper/MANIFEST.yaml).
- **A/B Compare** — two episodes on the **same seed** with synchronized
  playback (e.g. Greedy-GOMDP vs Adaptive AI): the governed side enforces 100%
  compliance and blocks injections while the ungoverned side does neither.
- **Shareable permalinks** — the Share button serializes all parameters into
  the URL; opening that URL rehydrates the exact configuration.
- **Command palette (⌘K / Ctrl-K)** — fuzzy jump to any screen, toggle
  theme/language, run the current episode, or fire a one-click preset
  (paper config, injection attack, Byzantine f=3, pretrained PPO).
- **Guided onboarding tour** — a first-visit walkthrough that steps through all
  six screens (switching the view behind it), reopenable from the “?” button.
- **GIF export** — server-side (matplotlib + imageio, PyroRL-style) episode GIF.
- **Arabic + English** with full RTL mirroring.
- **Dark + light** theme (system-aware, persisted, no flash).

> Tip: with verification enabled, fused confidence hovers near the τ boundary, so
> the **τ slider is a live lever** — lower τ (e.g. 0.70) to make alerts fire and
> watch the HITL gate / smart contract engage in the ledger.

---

## Quick start

Requires the repo's Python env (with `wildfire_governance` importable),
Node 18+, and the backend extras.

### Easiest — one command (auto-builds the UI)

```bash
pip install -r dashboard/backend/requirements.txt
python dashboard/run_dashboard.py --port 8123
```

The launcher builds the Next.js frontend on first run (the build output
`dashboard/frontend/out/` is git-ignored, so a fresh clone has none), then starts
the integrated server. Open **http://127.0.0.1:8123/**.

### Manual — explicit steps

```bash
# 1. Backend deps (in the repo's Python environment)
pip install -r dashboard/backend/requirements.txt

# 2. Build the frontend (produces dashboard/frontend/out/) — REQUIRED once
cd dashboard/frontend
npm install
npm run build
cd ../..

# 3. Run the integrated server (serves the UI + API + WebSocket)
python -m uvicorn dashboard.backend.main:app --host 127.0.0.1 --port 8123
```

Open **http://127.0.0.1:8123/**.

> **If you see `{"detail":"Not Found"}` or a "FRONTEND NOT BUILT" page at `/`,**
> the frontend hasn't been built yet — run step 2 (or use the one-command
> launcher above). The build directory is git-ignored, so it is never present in
> a fresh checkout.
>
> On Windows, ports in some reserved ranges (e.g. 8000) may be blocked with a
> "socket access forbidden" error — pick another port such as 8123.

---

## Security & Localhost-Only Notice

> [!WARNING]
> The dashboard is designed strictly for local demonstration and artifact validation:
> 1. **No Authentication:** The backend endpoints and WebSocket stream are completely unauthenticated.
> 2. **CORS Restrictions:** CORS is locked down specifically to `localhost:3000` and `127.0.0.1:3000`.
> 3. **Safe Binding:** Do not bind the server to `0.0.0.0` or expose the port to the public internet. Always run locally using the default `127.0.0.1` interface.

### Split dev mode (hot-reload frontend)

Run the API and the Next.js dev server separately:

```bash
# terminal 1 — API
python -m uvicorn dashboard.backend.main:app --port 8000 --reload

# terminal 2 — frontend (talks to the API via env vars)
cd dashboard/frontend
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

Open **http://localhost:3000/**. The backend already allows CORS from `:3000`.
When served from the FastAPI static mount instead, the UI auto-targets its own
origin, so no env vars are needed for the integrated build.

### Docker (one command)

```bash
docker compose -f dashboard/docker-compose.dashboard.yml up --build
# → http://localhost:8123
```

---

## Architecture

```
Browser (Next.js 14 + TS + Tailwind)
  ├─ GridCanvas ── decodes base64 heat/fire → inferno LUT → <canvas>
  ├─ ParameterPanel / PlaybackControls / MetricHUD / EventFeed
  └─ WebSocket ── /ws/simulate
        │
FastAPI (dashboard/backend) ── imports src/wildfire_governance
  ├─ simulation_service.stream_episode()  # mirrors runner.py, yields frames
  ├─ benchmark.run_benchmark()            # live multi-seed mean ± 95% CI
  ├─ render.episode_to_gif()              # imageio/matplotlib export
  └─ REST: /api/health /api/config/schema /api/methods /api/benchmark
           /api/export/gif /api/paper-results/{table}   (labeled reference)
```

### Files

| File | Role |
|---|---|
| `backend/schema.py` | Parameter schema + clamping + method→flags presets |
| `backend/simulation_service.py` | Live per-step frame generator (the core) |
| `backend/benchmark.py` | Live multi-seed aggregation with confidence intervals |
| `backend/render.py` | Server-side GIF export |
| `backend/main.py` | FastAPI app (REST + WebSocket + static UI mount) |
| `frontend/app/page.tsx` | Live Simulation screen |
| `frontend/lib/useEpisodeStream.ts` | WebSocket hook |
| `frontend/lib/colormap.ts` | Inferno LUT + base64 decode + fire-edge overlay |
| `frontend/components/sim/*` | Canvas, panel, playback, HUD, ledger |

---

## Scientific integrity

Per `Dashboard_Guide.md` §19, this dashboard computes results **live** from the
real simulation. It never presents the pre-committed `results/paper/*.csv` as a
live run. `/api/paper-results/{table}` exists only to show manuscript reference
values and always labels them as such. The compliance = 100% figure is enforced
*by construction* (Theorem 1), and the ledger/predicate view shows why.

Note on PPO movement: when the method requests PPO **and** the run uses the
checkpoint's dimensions (grid_size=100, N=20 UAVs, Z=25), the pretrained
`ppo_gomdp_best.pt` checkpoint loads and the real PPO policy drives UAV
movement (`policy_effective = "ppo_pretrained"`). For any other configuration
the viewer falls back to the greedy coordination policy and says so in the
summary — so `policy_effective` never misrepresents what ran. (In this
environment UAV positions do not affect fire dynamics or the global detection
signal, so PPO vs greedy changes the movement pattern, not L_d/F_p.)

---

## Status

| Phase | Screen / feature | State |
|---|---|---|
| 0 | Foundations (theme, i18n/RTL, tokens, API) | ✅ done |
| 1 | Live Simulation (canvas, params, playback, HUD, ledger, GIF) | ✅ done |
| 2 | Governance Explorer + Adversarial Lab (predicate inspector, validator ring, breach meter) | ✅ done |
| 3 | Benchmark + Reproducibility (live-vs-paper charts, honest diff, tradeoff frontier) | ✅ done |
| 4a | A/B split-screen + shareable permalinks | ✅ done |
| 4b | PPO checkpoint wiring (pretrained policy at grid=100, N=20) | ✅ done |
| 4c | VIIRS real-world screen + PyroRL-style renderer (green grid, warm slow fire, agent trails) | ✅ done |
| 4d | Command palette (⌘K) + guided onboarding tour | ✅ done |
| 4e | Paper-coverage screens (All Experiments, Ablation, Scalability, Learning, HITL, CNN) + consensus reference + supplementary badges | ✅ done |

Multi-view navigation shares one live run across screens (a single episode feeds
Live, Governance, and Adversarial). The backend endpoints for benchmarks,
paper-reference, and GIF export are already in place; the Phase 3 screens consume
them.

### API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/config/schema` | Parameter schema + method presets + colors |
| `GET /api/methods` | Method metadata |
| `WS /ws/simulate` | Live per-step episode stream |
| `POST /api/benchmark` | Live multi-seed mean ± 95% CI |
| `POST /api/export/gif` | Server-side episode GIF |
| `GET /api/breach-probability` | Theorem-2 breach curve (GOMDP vs centralized) |
| `GET /api/paper-results/{table}` | Manuscript reference values (labeled) for any canonical CSV |
| `GET /api/artifacts` | Enumerates every paper artifact with provenance class + CSV-present flag (drives the All-Experiments screen) |
