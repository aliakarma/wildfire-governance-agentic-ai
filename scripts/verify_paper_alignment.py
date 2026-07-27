#!/usr/bin/env python3
"""Assert that every committed result file matches the manuscript.

This is the submission gate. The expected values below are transcribed once,
by hand, from the tables and figures of Paper/AAAI/Wildfire.tex; the script then
checks each committed CSV under results/paper/ against them, cell by cell. If a
table in the manuscript changes and the results are not regenerated (or vice
versa), this fails and names the offending cell.

It is deliberately independent of scripts/generate_all_paper_results.py: the
aggregator derives the CSVs from per-seed data, this script checks the derived
values against the paper. Both agreeing is the evidence that the code and the
manuscript describe the same experiment.

    python scripts/verify_paper_alignment.py           # check results/paper
    python scripts/verify_paper_alignment.py --dir results/reproduced

Exit status 0 when everything matches, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

TOL = 0.05  # absolute, on values the manuscript prints to one decimal place
TOL_P = 0.001  # absolute, on p-values the manuscript prints to three decimals


# --------------------------------------------------------------------------- #
# Expected values, transcribed from Paper/AAAI/Wildfire.tex
# --------------------------------------------------------------------------- #

# Table 1 (tab:rl_comparison): method -> (L_d, L_d sd, F_p, F_p sd, FN_r, FN_r sd, compliance)
TABLE1 = {
    "PPO-GOMDP":    (15.1, 1.1,  6.0, 1.1, 2.1, 0.9, 100.0),
    "Greedy-GOMDP": (18.3, 1.4,  6.1, 1.3, 2.3, 1.0, 100.0),
    "Central+Sig":  (15.0, 1.1,  6.0, 1.2, None, None, 100.0),
    "Shield-PPO":   (15.2, 1.2,  6.2, 1.3, None, None, 100.0),
    "SafeLayer":    (14.9, 1.1,  7.0, 1.6, None, None,  98.4),
    "PPO-CMDP":     (14.8, 1.0,  8.3, 2.4, 2.6, 1.0,  92.8),
    "WCSAC":        (14.6, 1.2,  9.4, 2.0, 3.8, 1.4,  90.6),
    "Adaptive AI":  (16.2, 1.2, 22.4, 2.1, 0.9, 0.5,   0.0),
    "Static":       (41.5, 3.1, 15.3, 2.4, 1.8, 0.8,   0.0),
}

# Table 2 (tab:ablation): config -> (L_d, L_d sd, F_p, F_p sd, injections blocked)
TABLE2 = {
    "PPO-GOMDP (full)":         (15.1, 1.1,  6.0, 1.1, 100),
    "Greedy-GOMDP (full)":      (18.3, 1.4,  6.1, 1.3, 100),
    "- Adaptive coordination":  (29.7, 2.6,  6.1, 1.2, 100),
    "- HITL authorization":     (15.2, 1.1, 22.4, 2.2, 100),
    "- Consensus (Central+Sig)":(15.0, 1.1,  6.0, 1.2, 100),
    "- All authentication":     (15.1, 1.1,  6.9, 1.4,   0),
    "- Multi-stage verif.":     (15.0, 1.1, 14.8, 2.0, 100),
    "PPO-CMDP (no blockchain)": (14.8, 1.0,  8.3, 2.4,   0),
}

# Table 6 (tab:adversarial): (attack, parameter) -> (GOMDP, Central+Sig, Central)
TABLE3 = {
    ("No attack", "---"):                    (6.0, 6.0, 22.4),
    ("Spoofing (i.i.d.)", "p=0.05"):         (6.7, 6.7, 26.8),
    ("Spoofing (i.i.d.)", "p=0.10"):         (7.8, 7.9, 31.2),
    ("Spoofing (i.i.d.)", "p=0.20"):         (9.4, 9.5, 38.7),
    ("Spoofing (strategic)", "p=0.10"):      (8.6, 8.7, 34.5),
}
TABLE3_INJECTION = ("0/100", "0/100", "100/100")

# tab:realworld — VIIRS, per event and method: (L_d, L_d sd, F_p, F_p sd, compliance)
TABLE4 = {
    ("California '20", "PPO-GOMDP"):    (22.4, 3.2,  8.3, 2.1, 100.0),
    ("California '20", "Greedy-GOMDP"): (26.9, 3.8,  8.5, 2.3, 100.0),
    ("California '20", "PPO-CMDP"):     (22.0, 3.1, 10.6, 2.7,  93.1),
    ("California '20", "Adaptive AI"):  (20.1, 2.9, 24.6, 3.8,   0.0),
    ("Mediterranean '21", "PPO-GOMDP"):    (24.1, 4.1,  9.1, 2.5, 100.0),
    ("Mediterranean '21", "Greedy-GOMDP"): (28.8, 4.6,  9.3, 2.6, 100.0),
    ("Mediterranean '21", "PPO-CMDP"):     (23.6, 3.9, 11.4, 3.0,  92.4),
    ("Mediterranean '21", "Adaptive AI"):  (21.7, 3.5, 26.1, 4.2,   0.0),
    ("NSW '19–20", "PPO-GOMDP"):    (21.8, 2.7,  7.9, 1.9, 100.0),
    ("NSW '19–20", "Greedy-GOMDP"): (26.1, 3.3,  8.2, 2.1, 100.0),
    ("NSW '19–20", "PPO-CMDP"):     (21.3, 2.8, 10.1, 2.4,  93.6),
    ("NSW '19–20", "Adaptive AI"):  (19.8, 2.6, 23.9, 3.5,   0.0),
}

# tab:byzantine — stochastic theory (k=7, f=2) and deterministic empirical
BYZANTINE_THEORY = {0.05: (0.004, 0.05), 0.10: (0.026, 0.10),
                    0.20: (0.148, 0.20), 0.30: (0.353, 0.30)}
BYZANTINE_EMPIRICAL = {0: ("0/100", 6.0), 1: ("0/100", 6.1),
                       2: ("0/100", 6.2), 3: ("100/100", 8.9)}

# tab:ksweep — validator-count sweep at p_c = 0.10: k -> (f, theory, empirical)
#
# NOTE: the manuscript prints 0.007 for the k=13 theory cell. The exact binomial
# tail of Eq. (4) is 0.00646, which rounds to 0.006; the CSV carries the exact
# value and the manuscript cell was corrected to match.
KSWEEP = {4: (1, 0.052, 0.054), 7: (2, 0.026, 0.025),
          10: (3, 0.013, 0.013), 13: (4, 0.006, 0.006)}

# tab:hitl_sensitivity — p_err -> (FN_r, FN_r sd, F_p, F_p sd, compliance)
HITL = {
    0.05: (2.1, 0.9, 6.0, 1.1, 100.0),
    0.10: (3.9, 1.2, 5.5, 1.0, 100.0),
    0.15: (5.8, 1.5, 5.1, 0.9, 100.0),
    0.20: (7.8, 1.8, 4.2, 0.8, 100.0),
}

# tab:multisig
MULTISIG = (15.0, 1.1, 6.1, 1.2, 100, 100)

# Table 5 (tab:main_comparison) — the columns Table 1 does not carry
MAIN_EXTRA = {
    "ppo_gomdp":    (1.2, 3.0, -6.8),
    "greedy_gomdp": (1.2, 3.0, 13.0),
    "ppo_cmdp":     (None, 3.0, -8.6),
    "wcsac":        (None, None, -9.9),
    "adaptive_ai":  (None, None, None),
    "static":       (None, None, None),
}

# Sec. 6.2 + abstract — PPO-GOMDP vs PPO-CMDP detection latency.
# "t(19)=1.28, two-sided p=0.216 ... paired difference 0.3 steps (SE=0.234),
#  t_L(19)=5.55 (p<0.001) and t_U(19)=-2.99 (p=0.004), so max(p_L,p_U)=0.004"
STAT_TESTS = [
    ("paired t-test", {"statistic": 1.28, "p_value": 0.216}, "paired t"),
    ("TOST one-sided (lower", {"statistic": 5.55, "p_value": "<0.001"}, "TOST lower"),
    ("TOST one-sided (upper", {"statistic": -2.99, "p_value": 0.004}, "TOST upper"),
    ("TOST equivalence", {"p_value": 0.004}, "TOST max"),
]

# fig:learning (supporting the Sec. 5.2 convergence claim): validation L_d over
# training episodes, mean over 5 held-out seeds. Plateau at 15.1 by episode 650,
# stopping criterion met by episode 750.
LEARNING_CURVE = {0: 35.2, 100: 23.1, 200: 19.2, 300: 17.1, 400: 16.0,
                  500: 15.6, 600: 15.2, 700: 15.1, 800: 15.1}

# fig:falsealerts — F_p vs N
FIG_FP = {
    "PPO-GOMDP":    {5: 6.9, 10: 6.4, 20: 6.0, 40: 5.7},
    "Greedy-GOMDP": {5: 7.0, 10: 6.5, 20: 6.1, 40: 5.8},
    "PPO-CMDP":     {5: 9.8, 10: 9.1, 20: 8.3, 40: 7.6},
    "WCSAC":        {5: 10.2, 10: 9.6, 20: 9.4, 40: 8.7},
    "Adaptive AI":  {5: 23.8, 10: 23.1, 20: 22.4, 40: 21.9},
    "Static":       {5: 16.1, 10: 15.7, 20: 15.3, 40: 14.9},
}

# fig:latency — L_d vs N
FIG_LD = {
    "ppo_gomdp":    {5: 39.2, 10: 24.8, 20: 15.1, 40: 9.8},
    "greedy_gomdp": {5: 47.1, 10: 30.2, 20: 18.3, 40: 10.1},
    "ppo_cmdp":     {5: 38.8, 10: 24.4, 20: 14.8, 40: 9.6},
    "wcsac":        {5: 39.5, 10: 25.1, 20: 14.6, 40: 9.9},
    "adaptive_ai":  {5: 42.5, 10: 27.1, 20: 16.2, 40: 11.3},
    "static":       {5: 64.2, 10: 52.8, 20: 41.5, 40: 36.4},
}


# --------------------------------------------------------------------------- #
class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def num(self, where: str, expected, actual, tol: float = TOL) -> None:
        if expected is None:
            self.blank(where, actual)
            return
        self.checks += 1
        try:
            got = float(actual)
        except (TypeError, ValueError):
            self.failures.append(f"{where}: expected {expected}, got {actual!r}")
            return
        if abs(got - float(expected)) > tol:
            self.failures.append(f"{where}: expected {expected}, got {got}")

    def blank(self, where: str, actual) -> None:
        """The manuscript prints '---'; the CSV must be empty, not a number."""
        self.checks += 1
        if not (actual is None or (isinstance(actual, float) and pd.isna(actual))
                or str(actual).strip() in ("", "nan", "---")):
            self.failures.append(f"{where}: expected blank (manuscript '---'), got {actual!r}")

    def exact(self, where: str, expected, actual) -> None:
        self.checks += 1
        if str(actual).strip() != str(expected):
            self.failures.append(f"{where}: expected {expected!r}, got {actual!r}")


def load(base: Path, name: str) -> pd.DataFrame:
    path = base / name
    if not path.exists():
        raise SystemExit(f"[FAIL] missing result file: {path}")
    return pd.read_csv(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/paper", help="result directory to verify")
    args = ap.parse_args()
    base = Path(args.dir)
    r = Report()

    print(f"=== Paper alignment check: {base} vs Paper/AAAI/Wildfire.tex ===\n")

    # Table 1
    df = load(base, "table1_rl_comparison.csv").set_index("method")
    for m, (ld, lds, fp, fps, fn, fns, comp) in TABLE1.items():
        row = df.loc[m]
        r.num(f"Table 1 [{m}] L_d", ld, row.ld_mean)
        r.num(f"Table 1 [{m}] L_d sd", lds, row.ld_std)
        r.num(f"Table 1 [{m}] F_p", fp, row.fp_mean)
        r.num(f"Table 1 [{m}] F_p sd", fps, row.fp_std)
        r.num(f"Table 1 [{m}] FN_r", fn, row.fn_mean)
        r.num(f"Table 1 [{m}] FN_r sd", fns, row.fn_std)
        r.num(f"Table 1 [{m}] compliance", comp, row.compliance_pct)

    # Table 2
    df = load(base, "table2_ablation.csv").set_index("config")
    for c, (ld, lds, fp, fps, blocked) in TABLE2.items():
        row = df.loc[c]
        r.num(f"Table 2 [{c}] L_d", ld, row.ld_mean)
        r.num(f"Table 2 [{c}] L_d sd", lds, row.ld_std)
        r.num(f"Table 2 [{c}] F_p", fp, row.fp_mean)
        r.num(f"Table 2 [{c}] F_p sd", fps, row.fp_std)
        r.exact(f"Table 2 [{c}] injections", blocked, int(row.injections_blocked))
        r.exact(f"Table 2 [{c}] attempts", 100, int(row.injections_total))

    # Table 6 — adversarial
    df = load(base, "table3_adversarial.csv")
    keyed = df.set_index(["attack_type", "parameter"])
    for key, (g, s, c) in TABLE3.items():
        row = keyed.loc[key]
        r.num(f"Table 6 {key} GOMDP", g, row.gomdp)
        r.num(f"Table 6 {key} Cent.+Sig", s, row.central_sig)
        r.num(f"Table 6 {key} Central", c, row.central)
    inj = keyed.loc[("Alert injection (success)", "p_att=1")]
    for label, expected, actual in zip(
        ("GOMDP", "Cent.+Sig", "Central"), TABLE3_INJECTION,
        (inj.gomdp, inj.central_sig, inj.central),
    ):
        r.exact(f"Table 6 injection {label}", expected, actual)

    # VIIRS
    df = load(base, "table4_realworld_viirs.csv").set_index(["event", "method"])
    for key, (ld, lds, fp, fps, comp) in TABLE4.items():
        row = df.loc[key]
        r.num(f"VIIRS {key} L_d", ld, row.ld_mean)
        r.num(f"VIIRS {key} L_d sd", lds, row.ld_std)
        r.num(f"VIIRS {key} F_p", fp, row.fp_mean)
        r.num(f"VIIRS {key} F_p sd", fps, row.fp_std)
        r.num(f"VIIRS {key} compliance", comp, row.gov_compliance_pct)

    # Byzantine
    df = load(base, "table5_byzantine_theory.csv").set_index("p_c")
    for p_c, (gomdp, sig) in BYZANTINE_THEORY.items():
        row = df.loc[p_c]
        r.num(f"Byzantine theory p_c={p_c} GOMDP", gomdp, row.p_break_gomdp)
        r.num(f"Byzantine theory p_c={p_c} single verifier", sig, row.p_break_sig)
    df = load(base, "table5_byzantine_empirical.csv").set_index("f_c")
    for f_c, (breach, fp) in BYZANTINE_EMPIRICAL.items():
        row = df.loc[f_c]
        r.exact(f"Byzantine empirical f_c={f_c} breach", breach, row.breach)
        r.num(f"Byzantine empirical f_c={f_c} F_p", fp, row.fp_pct)

    # Validator-count sweep
    df = load(base, "table6_ksweep.csv").set_index("k")
    for k, (f, theory, emp) in KSWEEP.items():
        row = df.loc[k]
        r.exact(f"k-sweep k={k} f", f, int(row.f))
        r.num(f"k-sweep k={k} theory", theory, row.p_break_gomdp_theory)
        r.num(f"k-sweep k={k} empirical", emp, row.empirical)

    # HITL sensitivity
    df = load(base, "table7_hitl_sensitivity.csv").set_index("p_err")
    for p_err, (fn, fns, fp, fps, comp) in HITL.items():
        row = df.loc[p_err]
        r.num(f"HITL p_err={p_err} FN_r", fn, row.fn_mean)
        r.num(f"HITL p_err={p_err} FN_r sd", fns, row.fn_std)
        r.num(f"HITL p_err={p_err} F_p", fp, row.fp_mean)
        r.num(f"HITL p_err={p_err} F_p sd", fps, row.fp_std)
        r.num(f"HITL p_err={p_err} compliance", comp, row.gov_compliance_pct)

    # Multisig
    row = load(base, "table9_multisig.csv").iloc[0]
    ld, lds, fp, fps, blocked, total = MULTISIG
    r.num("Multisig L_d", ld, row.ld_mean)
    r.num("Multisig L_d sd", lds, row.ld_std)
    r.num("Multisig F_p", fp, row.fp_mean)
    r.num("Multisig F_p sd", fps, row.fp_std)
    r.exact("Multisig injections", blocked, int(row.injections_blocked))
    r.exact("Multisig attempts", total, int(row.injections_total))

    # Table 5 — full-metric extras
    df = load(base, "table1_rl_comparison_main.csv").set_index("config")
    for cfg, (bc, hr, overhead) in MAIN_EXTRA.items():
        row = df.loc[cfg]
        r.num(f"Table 5 [{cfg}] BC delay", bc, row.bc_delay_mean)
        r.num(f"Table 5 [{cfg}] human review", hr, row.human_review_mean)
        r.num(f"Table 5 [{cfg}] gov overhead", overhead, row.gov_overhead_pct)
        # L_d / F_p / FN_r / compliance must agree with Table 1
        alias = {"ppo_gomdp": "PPO-GOMDP", "greedy_gomdp": "Greedy-GOMDP",
                 "ppo_cmdp": "PPO-CMDP", "wcsac": "WCSAC",
                 "adaptive_ai": "Adaptive AI", "static": "Static"}[cfg]
        ld, lds, fp, fps, fn, fns, comp = TABLE1[alias]
        r.num(f"Table 5 [{cfg}] L_d", ld, row.ld_mean)
        r.num(f"Table 5 [{cfg}] F_p", fp, row.fp_mean)
        r.num(f"Table 5 [{cfg}] FN_r", fn, row.fn_mean)
        r.num(f"Table 5 [{cfg}] compliance", comp, row.compliance_pct)

    # Figures
    df = load(base, "fig2_false_alerts.csv").set_index(["config", "n_uavs"])
    for cfg, pts in FIG_FP.items():
        for n, fp in pts.items():
            r.num(f"Fig 2 [{cfg}] N={n}", fp, df.loc[(cfg, n)].fp_mean)
    df = load(base, "fig3_latency_data.csv").set_index(["config", "n_uavs"])
    for cfg, pts in FIG_LD.items():
        for n, ld in pts.items():
            r.num(f"Fig 4 [{cfg}] N={n}", ld, df.loc[(cfg, n)].ld_mean)

    # Statistical tests: the manuscript prints these explicitly in Sec. 6.2 and
    # the abstract, so every one is checked as a value, not just a threshold.
    st = load(base, "statistical_tests.csv")

    def stat_row(test_prefix):
        sub = st[(st.comparison == "PPO-GOMDP vs PPO-CMDP") & (st.metric == "L_d")
                 & st.test.str.startswith(test_prefix)]
        return None if sub.empty else sub.iloc[0]

    for prefix, expected, label in STAT_TESTS:
        row = stat_row(prefix)
        r.checks += 1
        if row is None:
            r.failures.append(f"statistical_tests.csv: missing row for '{prefix}'")
            continue
        for col, want in expected.items():
            if want == "<0.001":
                r.exact(f"stats [{label}] {col}", "<0.001", row[col])
            else:
                tol = TOL_P if col == "p_value" else TOL
                r.num(f"stats [{label}] {col}", want, row[col], tol)

    # Learning curve behind the Sec. 5.2 convergence claim.
    df = load(base, "fig3_learning_curve.csv").set_index("episode")
    for episode, ld in LEARNING_CURVE.items():
        r.num(f"Learning curve episode {episode} L_d", ld, df.loc[episode].ld_mean)
    r.num("Learning curve greedy baseline", 18.3, df.iloc[0].greedy_baseline)

    print(f"{r.checks} cells checked")
    if r.failures:
        print(f"\n{len(r.failures)} MISMATCH(ES):")
        for f in r.failures:
            print(f"  [FAIL] {f}")
        print("\nRESULT: FAIL — results and manuscript disagree")
        return 1
    print("\nRESULT: PASS — every committed value matches the manuscript")
    return 0


if __name__ == "__main__":
    sys.exit(main())
