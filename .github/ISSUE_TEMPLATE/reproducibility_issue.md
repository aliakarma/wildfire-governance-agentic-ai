---
name: Reproducibility Issue
about: Paper results don't reproduce within the stated 5% tolerance
title: "[REPRO] "
labels: reproducibility
---

## Paper Result
Which table or figure does not reproduce?

## Expected Value (from paper)
e.g., PPO-GOMDP Ld = 15.1 ± 1.1 steps (Table II)

## Observed Value
e.g., PPO-GOMDP Ld = 18.5 steps

## Environment
- OS:
- Python version:
- NumPy version (np.__version__):
- PyTorch version:
- Hardware: CPU model, RAM
- Docker or bare metal?

## Steps to Reproduce
```bash
# Paste exact commands
make reproduce-smoke
# OR
python experiments/01_main_comparison.py --config configs/experiments/paper_main_results.yaml --smoke
```

## Config Used
Paste the YAML config (or state: default paper_main_results.yaml).

## Notes
Results are expected to vary by up to 5% relative tolerance across platforms due to
floating-point differences in NumPy BLAS backends (MKL vs OpenBLAS).
Please check `scripts/check_reproducibility.sh` output before filing.
