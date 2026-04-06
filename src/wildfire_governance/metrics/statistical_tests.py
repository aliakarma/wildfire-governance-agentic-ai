"""Statistical tests with Holm-Bonferroni correction.

All comparisons in the paper use paired two-sided t-tests with
Holm-Bonferroni correction across all pairwise metric comparisons
to control familywise Type I error (Section VI-A).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats  # type: ignore[import]


@dataclass
class PairwiseTestResult:
    """Result of a single pairwise statistical test.

    Attributes:
        group_a: Name of comparison group A.
        group_b: Name of comparison group B.
        metric: Metric name (e.g. "ld", "fp").
        t_statistic: Paired t-test statistic.
        p_value_raw: Uncorrected p-value.
        p_value_corrected: Holm-Bonferroni corrected p-value.
        significant: True if p_value_corrected < alpha.
        mean_diff: mean(A) - mean(B).
        effect_size: Cohen's d.
    """

    group_a: str
    group_b: str
    metric: str
    t_statistic: float
    p_value_raw: float
    p_value_corrected: float
    significant: bool
    mean_diff: float
    effect_size: float


def paired_ttest_holm_bonferroni(
    comparisons: List[Tuple[str, str, str, List[float], List[float]]],
    alpha: float = 0.01,
) -> List[PairwiseTestResult]:
    """Run paired two-sided t-tests with Holm-Bonferroni correction.

    Args:
        comparisons: List of tuples (group_a_name, group_b_name, metric_name,
                     values_a, values_b). All value lists must have equal length.
        alpha: Familywise error rate (default 0.01, matching paper).

    Returns:
        List of PairwiseTestResult objects, one per comparison.

    Raises:
        ValueError: If any value list pair has unequal lengths.
    """
    raw_results = []
    for group_a, group_b, metric, vals_a, vals_b in comparisons:
        if len(vals_a) != len(vals_b):
            raise ValueError(
                f"Length mismatch for {group_a} vs {group_b} on {metric}: "
                f"{len(vals_a)} != {len(vals_b)}"
            )
        arr_a = np.array(vals_a, dtype=float)
        arr_b = np.array(vals_b, dtype=float)
        t_stat, p_raw = stats.ttest_rel(arr_a, arr_b)
        mean_diff = float(np.mean(arr_a) - np.mean(arr_b))
        pooled_std = float(
            np.sqrt(
                (np.std(arr_a, ddof=1) ** 2 + np.std(arr_b, ddof=1) ** 2) / 2.0
            )
        )
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0
        raw_results.append(
            (group_a, group_b, metric, float(t_stat), float(p_raw), mean_diff, effect_size)
        )

    # Holm-Bonferroni correction
    sorted_by_p = sorted(raw_results, key=lambda x: x[4])
    n = len(sorted_by_p)
    corrected_results = []
    for rank, (ga, gb, met, t_stat, p_raw, md, es) in enumerate(sorted_by_p):
        p_corr = min(1.0, p_raw * (n - rank))
        corrected_results.append(
            PairwiseTestResult(
                group_a=ga,
                group_b=gb,
                metric=met,
                t_statistic=t_stat,
                p_value_raw=p_raw,
                p_value_corrected=p_corr,
                significant=p_corr < alpha,
                mean_diff=md,
                effect_size=es,
            )
        )
    return corrected_results


def summarise_tests(results: List[PairwiseTestResult]) -> Dict:
    """Return a dict summary of all test results.

    Args:
        results: List of PairwiseTestResult from :func:`paired_ttest_holm_bonferroni`.

    Returns:
        Dict with keys ``n_tests``, ``n_significant``, ``all_significant``.
    """
    return {
        "n_tests": len(results),
        "n_significant": sum(1 for r in results if r.significant),
        "all_significant": all(r.significant for r in results),
        "results": [
            {
                "comparison": f"{r.group_a} vs {r.group_b} [{r.metric}]",
                "mean_diff": round(r.mean_diff, 4),
                "p_corrected": round(r.p_value_corrected, 6),
                "significant": r.significant,
                "effect_size": round(r.effect_size, 4),
            }
            for r in results
        ],
    }
