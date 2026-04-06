"""Experiment I/O utilities: save results, load paper CSVs, check reproducibility."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from wildfire_governance.utils.reproducibility import generate_run_hash


def save_results(
    results: list,
    output_dir: Path,
    filename: str,
) -> Path:
    """Save a list of dataclasses or dicts to a CSV file.

    Args:
        results: List of EpisodeResult dataclasses or plain dicts.
        output_dir: Directory to write the CSV into.
        filename: Output filename (e.g. ``"table3_main_comparison.csv"``).

    Returns:
        Path to the written CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    if not results:
        return out_path

    rows = [asdict(r) if hasattr(r, "__dataclass_fields__") else dict(r) for r in results]

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def aggregate_to_table(
    results: list,
    group_by: str = "config_name",
) -> pd.DataFrame:
    """Aggregate per-seed EpisodeResult list into a summary table.

    Args:
        results: List of EpisodeResult dataclasses.
        group_by: Column to group by (default: ``"config_name"``).

    Returns:
        DataFrame with mean ± std for ld, fp_pct, and governance_compliance_pct.
    """
    rows = [asdict(r) if hasattr(r, "__dataclass_fields__") else dict(r) for r in results]
    df = pd.DataFrame(rows)
    agg = (
        df.groupby(group_by)
        .agg(
            ld_mean=("ld", "mean"),
            ld_std=("ld", "std"),
            fp_mean=("fp_pct", "mean"),
            fp_std=("fp_pct", "std"),
            governance_compliance_pct=("governance_compliant", lambda x: x.mean() * 100.0),
            n_seeds=(group_by, "count"),
        )
        .reset_index()
    )
    return agg


def check_against_paper(
    run_csv: Path,
    paper_csv: Path,
    tolerance: float = 0.05,
    key_col: str = "config",
    metric_cols: list | None = None,
) -> Dict[str, bool]:
    """Compare run results to committed paper values within tolerance.

    Args:
        run_csv: Path to the newly generated results CSV.
        paper_csv: Path to the pre-committed paper results CSV.
        tolerance: Relative tolerance (default: 5%).
        key_col: Column used to align rows between the two files.
        metric_cols: Metric columns to compare. Defaults to [``"ld_mean"``, ``"fp_mean"``].

    Returns:
        Dict mapping ``"<config>/<metric>"`` → True if within tolerance.
    """
    if metric_cols is None:
        metric_cols = ["ld_mean", "fp_mean"]

    run_df = pd.read_csv(run_csv)
    paper_df = pd.read_csv(paper_csv)

    checks: Dict[str, bool] = {}
    for _, paper_row in paper_df.iterrows():
        key_val = paper_row[key_col]
        run_rows = run_df[run_df[key_col] == key_val]
        if run_rows.empty:
            checks[f"{key_val}/__missing__"] = False
            continue
        run_row = run_rows.iloc[0]
        for metric in metric_cols:
            if metric not in paper_row.index or metric not in run_row.index:
                continue
            paper_val = float(paper_row[metric])
            run_val = float(run_row[metric])
            if abs(paper_val) < 1e-9:
                within = abs(run_val) < tolerance
            else:
                within = abs(paper_val - run_val) / abs(paper_val) < tolerance
            checks[f"{key_val}/{metric}"] = within

    return checks
