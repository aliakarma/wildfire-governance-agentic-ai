"""Live episode streaming service.

``stream_episode`` runs the *real* wildfire simulation one timestep at a time
and yields a compact frame per step. It faithfully mirrors the governance
pipeline in ``experiments/utils/runner.py`` so that what the dashboard shows is
exactly what the experiments compute — no hardcoded numbers, no replayed CSVs.

Frame payload (JSON, one message per emitted step):
    {
      "type": "frame", "t": int, "grid_size": int,
      "heat_b64": str,          # uint8 grid*grid, value = heat*255
      "fire_b64": str,          # uint8 grid*grid, 0/1 ground-truth fire
      "uavs": [{"x": col, "y": row, "batt": frac}, ...],
      "event": {...} | null,    # alert approved/blocked/injection this step
      "metrics": {"ld", "fp_pct", "n_alerts", "n_false", "compliance"},
    }
Terminal message:
    {"type": "done", "summary": {...}, "ledger": [...], "meta": {...}}
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

# Make the repo's ``src`` importable regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wildfire_governance.agents.uav_agent import UAVAgent  # noqa: E402
from wildfire_governance.blockchain.consensus import (  # noqa: E402
    ByzantineFaultType,
    PBFTConsensus,
)
from wildfire_governance.blockchain.smart_contract import GovernanceSmartContract  # noqa: E402
from wildfire_governance.blockchain.transaction import build_transaction  # noqa: E402
from wildfire_governance.simulation.fire_propagation import FirePropagationConfig  # noqa: E402
from wildfire_governance.gomdp.invariant_checker import GovernanceInvariantChecker  # noqa: E402
from wildfire_governance.governance.hitl_interface import HITLAuthorisationGate  # noqa: E402
from wildfire_governance.governance.oracle_model import HumanOperatorOracle  # noqa: E402
from wildfire_governance.simulation.grid_environment import (  # noqa: E402
    EnvironmentConfig,
    WildfireGridEnvironment,
)
from wildfire_governance.utils.reproducibility import set_global_seed  # noqa: E402

from .schema import method_flags, validate_and_default
from .swarm import SwarmCoordinator

_MAX_FRAMES = 1200  # cap emitted frames; the sim still steps every timestep


def _b64_u8(arr01: np.ndarray) -> str:
    u8 = np.clip(np.asarray(arr01, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    return base64.b64encode(u8.tobytes()).decode("ascii")


def _b64_bits(mask: np.ndarray) -> str:
    u8 = (np.asarray(mask) > 0.5).astype(np.uint8)
    return base64.b64encode(u8.tobytes()).decode("ascii")


def _b64_fire(fire_age: np.ndarray) -> str:
    """Encode per-cell fire age (0 = grass, 1..255 = timesteps burning)."""
    u8 = np.minimum(np.asarray(fire_age), 255).astype(np.uint8)
    return base64.b64encode(u8.tobytes()).decode("ascii")


# Dashboard-only fire model: a slow, natural-looking front (avg P_spread ~0.014)
# vs the paper default of ~0.55. Kept deliberately gentle so a viewer can watch
# the UAVs search, verify and encircle the fire as it creeps outward through
# low-humidity corridors — a full 600-step episode burns ~12% of the grid.
# Affects only the live viewer, not the experiments.
_SLOW_FIRE = FirePropagationConfig(alpha1=0.10, alpha2=0.06, alpha3=8.6)


def _static_swarm(active: List[Any], sector_size: int, grid: int) -> Dict[str, Any]:
    """Fixed sector patrol for the ungoverned 'static' baseline (no coordination).

    Mirrors the old hardcoded targets so the baseline still looks uncoordinated
    next to the cooperative swarm used by every governed method.
    """
    targets = [((i * sector_size) % grid,
                ((i * sector_size) // grid * sector_size) % grid)
               for i in range(len(active))]
    return {"targets": targets, "roles": ["static"] * len(active),
            "links": [], "fires": [], "phase": "static"}


def _weather_index(obs: Dict[str, Any]) -> float:
    return float(
        np.clip(obs["wind_field"].mean() - obs["humidity_field"].mean() + 0.5, 0.0, 1.0)
    )


def _resolve_attack(params: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the UI ``attack_type`` enum into run_episode-style controls."""
    a = params["attack_type"]
    p_spoof = params["p_spoof"] if a in ("spoofing", "spoofing_strategic") else 0.0
    n_byz = params["n_byzantine"] if a == "byzantine" else 0
    return {"attack_type": a, "p_spoof": p_spoof, "n_byzantine": n_byz,
            "strategic": a == "spoofing_strategic"}


def stream_episode(
    raw_params: Dict[str, Any] | None, summary_only: bool = False
) -> Iterator[Dict[str, Any]]:
    """Yield per-step frames for one live episode, then a terminal summary.

    When ``summary_only`` is True, frame payloads (and their base64 encoding) are
    skipped entirely — used by the multi-seed benchmark for speed.
    """
    p = validate_and_default(raw_params)
    flags = method_flags(p["method"])
    atk = _resolve_attack(p)

    grid = p["grid_size"]
    n_uavs = p["n_uavs"]
    n_steps = p["n_timesteps"]
    tau = p["tau"]
    seed = p["seed"]

    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    env = WildfireGridEnvironment(EnvironmentConfig(
        grid_size=grid, n_timesteps=n_steps,
        n_ignition_points=2, fire_config=_SLOW_FIRE,
    ))
    obs = env.reset(seed=seed)
    fire_age = (obs["fire_mask"] > 0.5).astype(np.int32)

    fleet: List[UAVAgent] = [
        UAVAgent(
            agent_id=f"uav_{i}",
            initial_position=(int(rng.integers(0, grid)), int(rng.integers(0, grid))),
            grid_size=grid,
        )
        for i in range(n_uavs)
    ]
    # Sensor failure disables a prefix of the fleet.
    active = fleet
    if p["sensor_failure_rate"] > 0:
        n_failed = int(p["sensor_failure_rate"] * n_uavs)
        active = fleet[n_failed:] or fleet[:1]

    # Blockchain / governance components.
    consensus = PBFTConsensus(rng=rng)
    for i in range(min(atk["n_byzantine"], consensus.n_validators)):
        try:
            consensus.inject_byzantine_fault(i, ByzantineFaultType.MALICIOUS)
        except ValueError:
            pass

    # Contract honours the episode's tau so the confidence gate is consistent
    # end-to-end (the UI exposes tau as a live control).
    contract = GovernanceSmartContract(consensus=consensus, tau=tau) if flags["blockchain"] else None
    oracle = HumanOperatorOracle(rng=rng) if flags["hitl"] else None
    hitl = HITLAuthorisationGate(oracle=oracle, rng=rng) if flags["hitl"] else None
    checker = GovernanceInvariantChecker(tau=tau)

    spoofer = None
    if atk["p_spoof"] > 0:
        from wildfire_governance.adversarial.sensor_spoofer import SensorSpoofer
        spoofer = SensorSpoofer(p_spoof=atk["p_spoof"], rng=rng)

    # Coordination: the live viewer drives every governed/coordinated method
    # with a cooperative swarm controller (search → verify → encircle) so the
    # fleet visibly communicates and tracks the fire front. The "static"
    # baseline keeps a dumb fixed patrol. The method's greedy/PPO label still
    # describes its *governance* story; only the on-screen motion is unified.
    # (Visualization choice — the experiments/ pipeline is untouched.)
    coordinator = (
        SwarmCoordinator(grid_size=grid, n_uavs=len(active))
        if flags["coordination"] else None
    )
    sector_size = grid // max(1, int(np.sqrt(n_uavs)))
    policy_effective = flags["policy"]
    policy_note: Optional[str] = None

    # Trackers.
    ignition = 0
    first_detection: Optional[int] = None
    n_alerts = n_false = n_violations = 0
    n_inj_attempted = n_inj_blocked = 0
    trajectory: List[Dict[str, Any]] = []
    ledger: List[Dict[str, Any]] = []

    stride = max(1, -(-n_steps // _MAX_FRAMES))  # ceil division

    # Latest swarm-coordination output (targets/roles/links/fire overlays),
    # refreshed each simulated step and echoed into every emitted frame so the
    # canvas can draw who is talking to whom and where the fleet is heading.
    swarm: Dict[str, Any] = {
        "targets": [tuple(u.position) for u in active],
        "roles": ["scout" if coordinator else "static"] * len(active),
        "links": [], "fires": [],
        "phase": "search" if coordinator else "static",
    }

    def _uav_payload() -> List[Dict[str, Any]]:
        roles = swarm.get("roles") or []
        tgts = swarm.get("targets") or []
        out: List[Dict[str, Any]] = []
        for i, u in enumerate(active):
            d: Dict[str, Any] = {
                "x": int(u.position[1]), "y": int(u.position[0]),
                "batt": round(float(u.battery_fraction), 3),
                "role": roles[i] if i < len(roles) else "scout",
            }
            if i < len(tgts):
                tr, tc = tgts[i]
                d["tx"], d["ty"] = int(tc), int(tr)
            out.append(d)
        return out

    def _metrics() -> Dict[str, Any]:
        ld = (first_detection - ignition) if first_detection is not None else None
        fp = round(100.0 * n_false / max(1, n_alerts), 2)
        comp = round(100.0 * (1.0 - n_violations / max(1, n_alerts)), 2) if n_alerts else 100.0
        return {"ld": ld, "fp_pct": fp, "n_alerts": n_alerts, "n_false": n_false,
                "compliance": comp, "n_injections_blocked": n_inj_blocked}

    def _frame(t: int, event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {"type": "frame", "t": t, "grid_size": grid,
                "fire_b64": _b64_fire(fire_age), "uavs": _uav_payload(),
                "links": swarm.get("links", []), "fires": swarm.get("fires", []),
                "phase": swarm.get("phase", "search"), "event": event,
                "metrics": _metrics()}

    # Initial frame (t=0, reset state) so the canvas paints immediately.
    if not summary_only:
        yield _frame(0, None)

    for t in range(1, n_steps + 1):
        positions = [u.position for u in active]

        # Communication disruption: drop this step's sensing.
        if p["p_drop"] > 0 and rng.random() < p["p_drop"]:
            obs, done, _ = env.step(positions)
            fm = obs["fire_mask"] > 0.5
            fire_age[fm] += 1
            fire_age[~fm] = 0
            trajectory.append({"timestep": t, "alert_broadcast": False,
                               "governance_cert": None, "confidence": 0.0,
                               "human_approval": False})
            if done:
                break
            continue

        obs, done, sim_info = env.step(positions)
        fm = obs["fire_mask"] > 0.5
        fire_age[fm] += 1
        fire_age[~fm] = 0
        heat = obs["heat_map"].copy()
        if spoofer is not None:
            heat = spoofer.inject(heat, obs["fire_mask"], strategic=atk["strategic"])

        max_heat = float(heat.max())
        if max_heat > 0.60 and first_detection is None:
            first_detection = t

        # UAV movement. The cooperative swarm sees the current fire and the
        # pre-move fleet positions, then returns a target per UAV plus the comm
        # links and fire-cluster overlays for this frame. UAVs step one cell
        # toward their target (recharging if the battery is spent).
        if coordinator is not None:
            swarm = coordinator.step(obs["fire_mask"], list(positions))
        else:
            swarm = _static_swarm(active, sector_size, grid)
        for i, u in enumerate(active):
            try:
                u.move_to(swarm["targets"][i], rng)
            except Exception:
                u.recharge()

        step_info: Dict[str, Any] = {"timestep": t, "alert_broadcast": False,
                                     "governance_cert": None, "confidence": 0.0,
                                     "human_approval": False}
        event: Optional[Dict[str, Any]] = None

        if max_heat > 0.80 and first_detection is not None:
            weather = _weather_index(obs)
            conf = (float(np.clip(0.65 * max_heat + 0.35 * weather, 0.0, 1.0))
                    if flags["verification"] else max_heat)
            step_info["confidence"] = conf

            if conf > tau:
                r, c = np.unravel_index(heat.argmax(), heat.shape)
                is_true = bool(obs["fire_mask"][r, c] > 0.5)

                if flags["governance"] and flags["hitl"] and flags["blockchain"] and hitl and contract:
                    tx = build_transaction(
                        event_id=f"evt_{seed}_{t}",
                        geo_boundary=(int(r), int(c), int(r) + 1, int(c) + 1),
                        confidence_score=conf, sensor_readings={"heat": max_heat, "weather": weather})
                    decision, sig = hitl.process(tx, conf)
                    step_info["human_approval"] = decision.approved
                    if decision.approved and sig is not None:
                        res = contract.verify_and_execute(tx, sig, hitl.public_key_bytes)
                        predicate = {
                            "conf_ok": bool(res.confidence_ok),
                            "human_approval": True,
                            "signature_ok": bool(res.signature_ok),
                            "consensus_ok": bool(res.consensus_result.approved) if res.consensus_result else False,
                            "satisfied": bool(res.alert_enabled),
                        }
                        if res.alert_enabled:
                            n_alerts += 1
                            n_false += 0 if is_true else 1
                            step_info["alert_broadcast"] = True
                            step_info["governance_cert"] = res.cert
                            event = {"kind": "ALERT_APPROVED", "cert": res.cert[:12],
                                     "true_fire": is_true, "conf": round(conf, 3),
                                     "tau": tau, "row": int(r), "col": int(c),
                                     "predicate": predicate}
                        else:
                            event = {"kind": "ALERT_BLOCKED", "reason": res.contract_state.name,
                                     "conf": round(conf, 3), "tau": tau,
                                     "row": int(r), "col": int(c), "predicate": predicate}
                    else:
                        event = {"kind": "HITL_REJECTED", "conf": round(conf, 3), "tau": tau,
                                 "row": int(r), "col": int(c),
                                 "predicate": {"conf_ok": conf > tau, "human_approval": False,
                                               "signature_ok": None, "consensus_ok": None,
                                               "satisfied": False}}
                elif flags["governance"] and flags["hitl"] and not flags["blockchain"] and hitl:
                    decision, _ = hitl.process(tx := build_transaction(
                        event_id=f"evt_{seed}_{t}",
                        geo_boundary=(int(r), int(c), int(r) + 1, int(c) + 1),
                        confidence_score=conf, sensor_readings={"heat": max_heat}), conf)
                    step_info["human_approval"] = decision.approved
                    if decision.approved:
                        n_alerts += 1
                        n_false += 0 if is_true else 1
                        step_info["alert_broadcast"] = True  # signature-only: no on-chain cert
                        event = {"kind": "ALERT_SIGNED", "true_fire": is_true,
                                 "conf": round(conf, 3), "row": int(r), "col": int(c)}
                elif not flags["governance"]:
                    n_alerts += 1
                    n_false += 0 if is_true else 1
                    step_info["alert_broadcast"] = True
                    event = {"kind": "ALERT_UNGOVERNED", "true_fire": is_true,
                             "conf": round(conf, 3), "row": int(r), "col": int(c)}

                # Count a governance violation exactly as the invariant checker would.
                if step_info["alert_broadcast"]:
                    valid = (step_info["governance_cert"] is not None
                             and conf > tau and step_info["human_approval"])
                    if not valid:
                        n_violations += 1

                # Adversarial injection probe (blocked by construction under GOMDP).
                if t % 30 == 0 and flags["blockchain"] and contract:
                    n_inj_attempted += 1
                    blocked = not contract.attempt_unauthorised_injection(
                        (int(r), int(c), int(r) + 1, int(c) + 1))
                    if blocked:
                        n_inj_blocked += 1
                        event = {"kind": "INJECTION_BLOCKED", "row": int(r), "col": int(c)}

        # Dedicated injection attack schedule (independent of detection).
        if atk["attack_type"] == "injection" and t % 30 == 0 and flags["blockchain"] and contract:
            ar, ac = np.unravel_index(heat.argmax(), heat.shape)
            n_inj_attempted += 1
            if not contract.attempt_unauthorised_injection((int(ar), int(ac), int(ar) + 1, int(ac) + 1)):
                n_inj_blocked += 1
                event = {"kind": "INJECTION_BLOCKED", "row": int(ar), "col": int(ac)}

        trajectory.append(step_info)
        if event:
            ledger.append({"t": t, **event})

        if not summary_only and (t % stride == 0 or event is not None or done or t == n_steps):
            yield _frame(t, event)
        if done:
            break

    report = checker.check_trajectory(trajectory)
    ld = first_detection - ignition if first_detection is not None else None
    yield {
        "type": "done",
        "summary": {
            "ld": ld,
            "fp_pct": round(100.0 * n_false / max(1, n_alerts), 2),
            "compliance": 100.0 if report.theorem1_satisfied else round(report.compliance_rate * 100, 2),
            "n_alerts": n_alerts, "n_false": n_false,
            "n_injections_attempted": n_inj_attempted,
            "n_injections_blocked": n_inj_blocked,
            "theorem1_satisfied": bool(report.theorem1_satisfied),
        },
        "ledger": ledger[-200:],  # cap payload
        "meta": {
            "method": p["method"], "method_label": flags["label"],
            "enforcement": flags["enforcement"],
            "policy_requested": flags["policy"], "policy_effective": policy_effective,
            "n_validators": int(consensus.n_validators),
            "n_byzantine": int(atk["n_byzantine"]),
            "byzantine_threshold": int((consensus.n_validators - 1) // 3),
            "params": p,
            "note": policy_note,
        },
    }


def collect_episode(raw_params: Dict[str, Any] | None) -> Dict[str, Any]:
    """Run an episode to completion, returning frames + terminal payload.

    Used by the GIF exporter and by synchronous callers/tests.
    """
    frames: List[Dict[str, Any]] = []
    done: Dict[str, Any] = {}
    for msg in stream_episode(raw_params):
        if msg["type"] == "frame":
            frames.append(msg)
        else:
            done = msg
    return {"frames": frames, "done": done}


def summarize_episode(raw_params: Dict[str, Any] | None) -> Dict[str, Any]:
    """Run an episode and return only the terminal summary (frames discarded).

    Used by the multi-seed benchmark so method differences in F_p / compliance
    surface (this engine honours tau end-to-end, unlike the 0.80-hardcoded
    alert gate in experiments/utils/runner.py)."""
    done: Dict[str, Any] = {}
    for msg in stream_episode(raw_params, summary_only=True):
        if msg["type"] == "done":
            done = msg
    return done
