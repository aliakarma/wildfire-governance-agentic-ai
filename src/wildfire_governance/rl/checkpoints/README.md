# PPO-GOMDP Checkpoints

Training writes `ppo_gomdp_best.pt` (best validation reward) and
`ppo_gomdp_final.pt` (last episode) into this directory.

**The supplementary archive ships without them.** `ppo_gomdp_best.pt` is ~63 MB,
and a PyTorch checkpoint is an internal zip so it does not compress further;
including it would consume most of the 100 MB submission budget. Nothing on the
paper-verification path needs one — see [TRAINING.md](../../../../TRAINING.md).

## Training configuration

| Parameter | Value |
|-----------|-------|
| Python | 3.10.14 |
| PyTorch | 2.2.1 |
| Grid size | 100 × 100 |
| UAV fleet (N) | 20 |
| Sectors (Z) | 25 |
| Training episodes | 1000 |
| Random seed | 42 |

## Expected evaluation metrics (Table 1)

| Metric | Value |
|--------|-------|
| Ld (mean ± std) | 15.1 ± 1.1 steps |
| Fp (mean ± std) | 6.0 ± 1.1% |
| Governance compliance | 100.0% |

## Producing one

```bash
# Bash (~4 hours on 8 CPU cores)
make train-ppo

# PowerShell
python experiments/11_ppo_training.py --config configs/experiments/ppo_training.yaml

# Quick smoke test (2 episodes)
python experiments/11_ppo_training.py --config configs/experiments/ppo_training.yaml --smoke
```

## Verifying one

```bash
# Bash
make eval-ppo

# PowerShell
python -m wildfire_governance.rl.evaluator --use_pretrained --n_seeds 5
```

A checkpoint loads only when its tensor shapes match the agent being evaluated,
so grid size and fleet size must match those it was trained with (100 × 100,
N = 20). Evaluation scripts stop with an explanatory message rather than falling
back to random weights; pass `--allow-untrained` to override deliberately.

Checkpoints from revisions predating the fused policy head are migrated
automatically on load — see the last section of
[TRAINING.md](../../../../TRAINING.md).
