"""Publication-quality matplotlib helpers aligned with IEEE TII figure style."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- IEEE TII column widths (inches) ---
SINGLE_COL = 3.5
DOUBLE_COL = 7.16

# --- Consistent colour/marker palette ---
PALETTE: dict[str, str] = {
    "ppo_gomdp": "#1f77b4",
    "greedy_gomdp": "#ff7f0e",
    "ppo_cmdp": "#9467bd",
    "adaptive_ai": "#2ca02c",
    "static": "#d62728",
    "proposed": "#1f77b4",
}
MARKERS: dict[str, str] = {
    "ppo_gomdp": "o",
    "greedy_gomdp": "s",
    "ppo_cmdp": "D",
    "adaptive_ai": "^",
    "static": "v",
    "proposed": "o",
}

_STYLE_APPLIED = False


def apply_ieee_style() -> None:
    """Apply IEEE TII-compatible matplotlib rcParams globally."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
        }
    )
    _STYLE_APPLIED = True


def plot_latency_vs_uavs(
    data: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot detection latency Ld vs. UAV fleet size N (Fig. 3).

    Args:
        data: DataFrame with columns ``config``, ``n_uavs``, ``ld_mean``, ``ld_std``.
        output_path: If given, saves the figure to this path.
        show: Whether to call ``plt.show()``.

    Returns:
        Matplotlib Figure object.
    """
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.8))
    configs = data["config"].unique()
    for cfg in configs:
        sub = data[data["config"] == cfg].sort_values("n_uavs")
        color = PALETTE.get(cfg, "gray")
        marker = MARKERS.get(cfg, "o")
        label = _label(cfg)
        ax.errorbar(
            sub["n_uavs"],
            sub["ld_mean"],
            yerr=sub["ld_std"],
            label=label,
            color=color,
            marker=marker,
            capsize=3,
        )
    ax.set_xlabel("Number of UAVs ($N$)")
    ax.set_ylabel("Detection Latency $L_d$ (steps)")
    ax.legend(fontsize=8)
    if output_path:
        fig.savefig(output_path)
    if show:
        plt.show()
    return fig


def plot_false_alert_bar(
    data: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot false alert rate Fp bar chart (Fig. 4).

    Args:
        data: DataFrame with columns ``config``, ``fp_mean``, ``fp_std``.
        output_path: If given, saves the figure.
        show: Whether to call ``plt.show()``.

    Returns:
        Matplotlib Figure object.
    """
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    x = np.arange(len(data))
    colors = [PALETTE.get(c, "steelblue") for c in data["config"]]
    ax.bar(x, data["fp_mean"], yerr=data["fp_std"], color=colors, capsize=4, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(c) for c in data["config"]], fontsize=8)
    ax.set_ylabel("False Alert Rate $F_p$ (%)")
    if output_path:
        fig.savefig(output_path)
    if show:
        plt.show()
    return fig


def plot_tradeoff_frontier(
    data: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot latency vs. false-alert Pareto frontier (Fig. 5, N=40 UAVs).

    Args:
        data: DataFrame with columns ``config``, ``ld_mean``, ``ld_std``,
              ``fp_mean``, ``fp_std``.
        output_path: If given, saves the figure.
        show: Whether to call ``plt.show()``.

    Returns:
        Matplotlib Figure object.
    """
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    for _, row in data.iterrows():
        cfg = row["config"]
        ax.errorbar(
            row["ld_mean"],
            row["fp_mean"],
            xerr=row.get("ld_std", 0),
            yerr=row.get("fp_std", 0),
            fmt=MARKERS.get(cfg, "o"),
            color=PALETTE.get(cfg, "gray"),
            label=_label(cfg),
            capsize=3,
            markersize=7,
        )
    ax.set_xlabel("Detection Latency $L_d$ (steps)")
    ax.set_ylabel("False Alert Rate $F_p$ (%)")
    ax.set_title("Latency–False Alert Frontier ($N=40$)", fontsize=9)
    ax.legend(fontsize=8)
    if output_path:
        fig.savefig(output_path)
    if show:
        plt.show()
    return fig


def _label(config_key: str) -> str:
    """Convert a config key to a display label."""
    mapping = {
        "ppo_gomdp": "PPO-GOMDP",
        "greedy_gomdp": "Greedy-GOMDP",
        "ppo_cmdp": "PPO-CMDP",
        "adaptive_ai": "Adaptive AI",
        "static": "Static",
        "proposed": "Proposed",
    }
    return mapping.get(config_key, config_key.replace("_", " ").title())
