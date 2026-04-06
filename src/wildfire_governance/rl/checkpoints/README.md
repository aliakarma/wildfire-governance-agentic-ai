# PPO-GOMDP Checkpoints

## Pre-trained Checkpoint: `ppo_gomdp_best.pt`

Trained with the configuration in `configs/experiments/ppo_training.yaml`.

| Parameter | Value |
|-----------|-------|
| Python | 3.10.14 |
| PyTorch | 2.2.1 |
| Grid size | 100 × 100 |
| UAV fleet (N) | 20 |
| Sectors (Z) | 25 |
| Training episodes | 1000 |
| Random seed | 42 |

### Expected evaluation metrics (Table II in paper)

| Metric | Value |
|--------|-------|
| Ld (mean ± std) | 15.1 ± 1.1 steps |
| Fp (mean ± std) | 6.0 ± 1.1% |
| Governance compliance | 100.0% |

### Verification

```bash
# Bash
make eval-ppo

# PowerShell
python src/wildfire_governance/rl/evaluator.py --use_pretrained --n_seeds 5 --smoke
```

### Re-training from scratch

```bash
# Bash (~4 hours on 8 CPU cores)
make train-ppo

# PowerShell
python experiments/11_ppo_training.py --config configs/experiments/ppo_training.yaml

# Quick smoke test (2 episodes)
python experiments/11_ppo_training.py --config configs/experiments/ppo_training.yaml --smoke
```

Note: The checkpoint file `ppo_gomdp_best.pt` is tracked in git via
`.gitignore` exception rules. It is approximately 2.5 MB and contains
the policy network, value network, and optimiser state.
