"""Experiment I/O utilities."""
from __future__ import annotations
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

def save_results(results, output_dir, filename):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    if not results: return out_path
    rows = [asdict(r) if hasattr(r, "__dataclass_fields__") else dict(r) for r in results]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    return out_path

def aggregate_to_table(results, group_by="config_name"):
    rows = [asdict(r) if hasattr(r, "__dataclass_fields__") else dict(r) for r in results]
    df = pd.DataFrame(rows)
    agg = df.groupby(group_by).agg(
        ld_mean=("ld","mean"), ld_std=("ld","std"),
        fp_mean=("fp_pct","mean"), fp_std=("fp_pct","std"),
        governance_compliance_pct=("governance_compliant", lambda x: x.mean()*100.0),
        n_seeds=(group_by,"count")).reset_index()
    return agg
