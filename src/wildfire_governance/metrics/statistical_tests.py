"""Statistical tests with Holm-Bonferroni correction."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats

@dataclass
class PairwiseTestResult:
    group_a: str; group_b: str; metric: str; t_statistic: float
    p_value_raw: float; p_value_corrected: float; significant: bool
    mean_diff: float; effect_size: float

def paired_ttest_holm_bonferroni(comparisons, alpha=0.01):
    raw_results = []
    for ga, gb, metric, va, vb in comparisons:
        if len(va) != len(vb): raise ValueError(f"Length mismatch for {ga} vs {gb} on {metric}")
        aa, ab = np.array(va, dtype=float), np.array(vb, dtype=float)
        t_stat, p_raw = stats.ttest_rel(aa, ab); md = float(np.mean(aa)-np.mean(ab))
        ps = float(np.sqrt((np.std(aa,ddof=1)**2+np.std(ab,ddof=1)**2)/2.0))
        es = md/ps if ps > 0 else 0.0
        raw_results.append((ga, gb, metric, float(t_stat), float(p_raw), md, es))
    sorted_r = sorted(raw_results, key=lambda x: x[4]); n = len(sorted_r)
    return [PairwiseTestResult(ga,gb,met,t,p,min(1.0,p*(n-rank)),min(1.0,p*(n-rank))<alpha,md,es)
            for rank,(ga,gb,met,t,p,md,es) in enumerate(sorted_r)]

def summarise_tests(results):
    return {"n_tests": len(results), "n_significant": sum(1 for r in results if r.significant),
            "all_significant": all(r.significant for r in results),
            "results": [{"comparison": f"{r.group_a} vs {r.group_b} [{r.metric}]",
                         "mean_diff": round(r.mean_diff,4), "p_corrected": round(r.p_value_corrected,6),
                         "significant": r.significant, "effect_size": round(r.effect_size,4)} for r in results]}
