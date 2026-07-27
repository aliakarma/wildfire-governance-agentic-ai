#!/usr/bin/env python3
"""Experiment 11c — parallel multi-seed PPO-GOMDP training.

Trains one PPO-GOMDP policy per seed, with seeds running concurrently as
separate processes sharing a single GPU. Produces, per seed, a checkpoint and a
learning curve, plus an aggregate mean/std curve across seeds for the
manuscript's training figure.

Why this exists alongside 11_ppo_training.py:
  * **Parallel seeds.** The paper reports a validation curve over multiple
    seeds; running them sequentially is N times slower for no reason.
  * **Incremental writes.** 11_ppo_training.py writes its learning curve only
    after the final episode, so an interrupted run loses its entire history.
    This script appends after every episode, so a run killed at any point still
    yields usable data.
  * **Device awareness.** Rollout is ~72% policy-forward wall-clock, so the
    accelerator matters; PPOGOMDPAgent now honours `device`.

Usage:
    python experiments/11c_train_multiseed.py --seeds 5 --episodes 1000
    python experiments/11c_train_multiseed.py --seeds 5 --workers 5 --smoke

Outputs (under --outdir, default results/runs/multiseed_<timestamp>/):
    seed_<k>/ppo_learning_curve.csv     per-episode reward / L_d / compliance
    seed_<k>/ppo_gomdp_seed<k>.pt       per-seed checkpoint (best reward)
    seed_<k>/status.json                live progress, safe to poll
    learning_curve_aggregate.csv        mean/std across seeds per episode
    best_checkpoint.pt                  copy of the best seed's checkpoint
    summary.json                        per-seed finals + wall-clock
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

# Paper's PPO learning rate, annealed linearly to zero over training.
_LR0 = 3e-4

# Episodes averaged when deciding whether to checkpoint. Selecting on a single
# episode's reward picks the luckiest ignition draw rather than the best
# policy: in the 500-episode run the two seeds' best single episodes differed
# by 0.0004, which is noise, while their converged L_d differed by 62 steps.
_CKPT_WINDOW = 25


def _train_one_seed(args: dict) -> dict:
    """Train a single seed to completion. Runs in its own process."""
    seed = args["seed"]
    n_episodes = args["n_episodes"]
    grid_size = args["grid_size"]
    n_timesteps = args["n_timesteps"]
    n_uavs = args["n_uavs"]
    device = args["device"]
    seed_dir = Path(args["seed_dir"])
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Keep per-process thread pools small: with W workers on one machine,
    # letting every process grab all cores causes thrashing, not speedup.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import numpy as np
    import torch
    torch.set_num_threads(1)

    from wildfire_governance.rl.gomdp_env import GOMMDPGymEnv
    from wildfire_governance.rl.ppo_agent import PPOGOMDPAgent
    from wildfire_governance.gomdp.invariant_checker import GovernanceInvariantChecker
    from wildfire_governance.simulation.grid_environment import EnvironmentConfig

    env = GOMMDPGymEnv(
        config=EnvironmentConfig(grid_size=grid_size, n_timesteps=n_timesteps),
        n_uavs=n_uavs,
        enable_governance=True,
    )
    agent = PPOGOMDPAgent(
        grid_size=grid_size, n_uavs=n_uavs, lr=_LR0, clip_ratio=0.2,
        entropy_coeff=0.01, gamma=0.99, n_epochs=4, device=device,
    )
    checker = GovernanceInvariantChecker(tau=0.80)

    curve_path = seed_dir / "ppo_learning_curve.csv"
    status_path = seed_dir / "status.json"
    ckpt_path = seed_dir / f"ppo_gomdp_seed{seed}.pt"

    # Header written once; rows appended per episode so an interrupted run
    # still leaves a usable history on disk.
    with curve_path.open("w", encoding="utf-8") as fh:
        fh.write("episode,reward,ld,compliance,fp_pct,n_alerts,loss,elapsed_s\n")

    best_reward = -float("inf")
    best_rolling_ld = float("inf")
    recent_lds: list[float] = []
    t_start = time.time()

    for ep in range(n_episodes):
        # Linear anneal (Schulman et al. 2017). Held fixed, the run peaked near
        # episode 375 of 500 and then gave back 21% of its L_d gain — a step
        # size that is right early and too large to settle with later.
        agent.set_learning_rate(_LR0 * (1.0 - ep / n_episodes))
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], []
        done = False
        total_reward = 0.0
        info: dict = {}

        while not done:
            action_dict = agent.select_actions(obs, env._fleet)
            action_arr = np.array([action_dict.get(i, 0) for i in range(n_uavs)])
            next_obs, reward, terminated, truncated, info = env.step(action_arr)

            ep_obs.append(obs.copy() if hasattr(obs, "copy") else obs)
            ep_actions.append(action_dict)
            ep_rewards.append(float(reward))
            ep_dones.append(bool(terminated or truncated))

            obs = next_obs
            total_reward += float(reward)
            done = terminated or truncated

        loss = agent.update(ep_obs, ep_actions, ep_rewards, ep_dones)
        report = checker.check_trajectory(env.get_trajectory())
        ep_ld = float(info.get("episode_ld", float("inf")))
        compliance = float(report.compliance_rate)
        elapsed = time.time() - t_start

        # F_p is the metric the false-alert-suppression claim rests on, so it
        # belongs in the curve rather than being recoverable only by re-running.
        fp_pct = float(info.get("episode_fp_pct", 0.0))
        n_alerts = int(getattr(env, "_n_alerts_broadcast", 0))

        with curve_path.open("a", encoding="utf-8") as fh:
            ld_str = "" if not np.isfinite(ep_ld) else f"{ep_ld:.4f}"
            fh.write(f"{ep},{total_reward:.6f},{ld_str},"
                     f"{compliance:.6f},{fp_pct:.4f},{n_alerts},"
                     f"{loss:.6f},{elapsed:.2f}\n")

        best_reward = max(best_reward, total_reward)

        # Checkpoint on a rolling mean of L_d — the metric the paper reports —
        # so the saved policy is one that performs consistently rather than one
        # that caught a single favourable ignition. Episodes that never detect
        # count as the full horizon rather than being dropped, otherwise a
        # policy that finds nothing would score as though it found everything.
        recent_lds.append(ep_ld if np.isfinite(ep_ld) else float(n_timesteps))
        if len(recent_lds) > _CKPT_WINDOW:
            recent_lds.pop(0)
        rolling_ld = float(np.mean(recent_lds))

        if ep == 0:
            agent.save_checkpoint(str(ckpt_path))  # ensure the file always exists
        elif len(recent_lds) == _CKPT_WINDOW and rolling_ld < best_rolling_ld:
            best_rolling_ld = rolling_ld
            agent.save_checkpoint(str(ckpt_path))

        status_path.write_text(json.dumps({
            "seed": seed,
            "episode": ep + 1,
            "n_episodes": n_episodes,
            "pct": round(100.0 * (ep + 1) / n_episodes, 2),
            "best_reward": best_reward,
            "rolling_ld": round(rolling_ld, 1),
            "best_rolling_ld": (
                round(best_rolling_ld, 1) if np.isfinite(best_rolling_ld) else None
            ),
            "last_ld": ep_ld if np.isfinite(ep_ld) else None,
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(elapsed / (ep + 1) * (n_episodes - ep - 1), 1),
        }), encoding="utf-8")

    return {
        "seed": seed,
        "best_reward": best_reward,
        "best_rolling_ld": (
            best_rolling_ld if np.isfinite(best_rolling_ld) else None
        ),
        "episodes": n_episodes,
        "wall_s": round(time.time() - t_start, 1),
        "checkpoint": str(ckpt_path),
        "curve": str(curve_path),
    }


def aggregate(outdir: Path, seeds: list[int]) -> None:
    """Aggregate per-seed curves into mean/std per episode."""
    import numpy as np
    import pandas as pd

    frames = []
    for s in seeds:
        p = outdir / f"seed_{s}" / "ppo_learning_curve.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["seed"] = s
            frames.append(df)
    if not frames:
        print("no per-seed curves found; nothing to aggregate")
        return

    allc = pd.concat(frames, ignore_index=True)
    agg_spec = dict(
        reward_mean=("reward", "mean"), reward_std=("reward", "std"),
        ld_mean=("ld", "mean"), ld_std=("ld", "std"),
        compliance_mean=("compliance", "mean"),
        n_seeds=("seed", "nunique"),
    )
    if "fp_pct" in allc.columns:
        agg_spec["fp_pct_mean"] = ("fp_pct", "mean")
    grouped = allc.groupby("episode").agg(**agg_spec).reset_index()
    out = outdir / "learning_curve_aggregate.csv"
    grouped.to_csv(out, index=False)
    print(f"wrote {out}  ({len(grouped)} episodes x {allc['seed'].nunique()} seeds)")

    tail = grouped.tail(50)
    print("\n=== converged (mean over last 50 episodes) ===")
    print(f"  L_d        : {tail['ld_mean'].mean():.2f}")
    print(f"  reward     : {tail['reward_mean'].mean():.2f}")
    print(f"  compliance : {100 * grouped['compliance_mean'].mean():.1f}%")
    if "fp_pct_mean" in grouped.columns:
        print(f"  F_p        : {tail['fp_pct_mean'].mean():.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (0..N-1)")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel processes (0 = min(seeds, cpu_count))")
    ap.add_argument("--grid", type=int, default=100)
    ap.add_argument("--timesteps", type=int, default=3000)
    ap.add_argument("--uavs", type=int, default=20)
    ap.add_argument("--device", default=None, help="cuda | cpu | None=auto")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.episodes = 2, 3
        args.grid, args.timesteps, args.uavs = 40, 300, 8

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else REPO / "results" / "runs" / f"multiseed_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        device, gpu = "cpu", "none"

    workers = args.workers or min(args.seeds, mp.cpu_count())
    seeds = list(range(args.seeds))

    print("=" * 62)
    print("PPO-GOMDP multi-seed training")
    print("=" * 62)
    print(f"  seeds      : {args.seeds}  (parallel workers: {workers})")
    print(f"  episodes   : {args.episodes} per seed")
    print(f"  env        : grid {args.grid}, {args.timesteps} steps, {args.uavs} UAVs")
    print(f"  device     : {device}   GPU: {gpu}")
    print(f"  outdir     : {outdir}")
    print("=" * 62, flush=True)

    jobs = [{
        "seed": s, "n_episodes": args.episodes, "grid_size": args.grid,
        "n_timesteps": args.timesteps, "n_uavs": args.uavs, "device": device,
        "seed_dir": str(outdir / f"seed_{s}"),
    } for s in seeds]

    t0 = time.time()
    # 'spawn' is required for CUDA in child processes; fork corrupts the context.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        results = pool.map(_train_one_seed, jobs)
    wall = time.time() - t0

    # Rank by converged detection latency, not by best single-episode reward:
    # the latter separated two seeds by 0.0004 (noise) while their L_d differed
    # by 62 steps, so it chose the exported checkpoint essentially at random.
    results = sorted(
        results,
        key=lambda r: (r["best_rolling_ld"] if r["best_rolling_ld"] is not None
                       else float("inf")),
    )
    best = results[0]
    shutil.copy2(best["checkpoint"], outdir / "best_checkpoint.pt")

    (outdir / "summary.json").write_text(json.dumps({
        "seeds": args.seeds, "episodes": args.episodes, "workers": workers,
        "device": device, "gpu": gpu,
        "wall_clock_s": round(wall, 1),
        "wall_clock_h": round(wall / 3600, 2),
        "best_seed": best["seed"],
        "results": results,
    }, indent=2), encoding="utf-8")

    print(f"\nall seeds done in {wall/3600:.2f} h "
          f"({wall/max(args.seeds*args.episodes,1):.2f} s/episode effective)")
    print(f"best seed: {best['seed']}  (rolling L_d {best['best_rolling_ld']}, "
          f"best reward {best['best_reward']:.3f})")
    aggregate(outdir, seeds)
    print(f"\noutputs in {outdir}")


if __name__ == "__main__":
    main()
