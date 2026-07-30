# Training the PPO-GOMDP policy

**This archive does not ship a trained checkpoint.** `ppo_gomdp_best.pt` is 63 MB
— a PyTorch checkpoint is an internal zip, so it barely compresses — and the
submission limit is 100 MB. Excluding it takes the archive from ~62 MB to ~4 MB.

**You do not need a checkpoint to verify the paper.** Every headline claim is
verifiable from the committed results. Training is required only if you want to
re-derive the learned policy yourself.

---

## What you can verify without training

| Command | Verifies | Time |
| :--- | :--- | :--- |
| `python scripts/verify_paper_alignment.py` | All 354 manuscript cells, table by table | ~30 s |
| `python -m pytest tests/smoke/ -v --no-cov` | GOMDP invariant, Ed25519 signing, fire propagation | ~10 s |
| `python -m pytest tests/ --no-cov` | Full suite (87 tests) | ~15 s |
| `python dashboard/run_dashboard.py` | Live simulation, governance predicate, adversarial lab, A/B compare | instant |
| `python experiments/13_byzantine_compromise.py` | Table 8 — validator compromise (closed form) | ~1 min |
| `python experiments/14_ksweep.py` | Table 9 — validator sweep (closed form) | ~1 min |
| `python experiments/16_multisig.py` | Table 11 — *m*-of-*n* multisignature | ~1 min |

The dashboard needs neither a checkpoint nor Node.js: the UI ships pre-built in
`dashboard/frontend/out/`, and the backend's search policy is analytic.

Experiments that **do** need a checkpoint — Table 1 (`11b_rl_comparison.py`),
Table 5 (`01_main_comparison.py`), and Table 7 VIIRS (`08*_viirs_*.py`) — stop
with an explanatory message rather than silently running an untrained policy.

---

## Train it

### Single seed — reproduces the packaged checkpoint

```bash
python experiments/11_ppo_training.py --config configs/experiments/ppo_training.yaml
```

Writes `src/wildfire_governance/rl/checkpoints/ppo_gomdp_best.pt`.

**~4 hours on 8 CPU cores.** Check the training loop first (~2 minutes):

```bash
python experiments/11_ppo_training.py --config configs/experiments/ppo_training.yaml --smoke
```

Hyperparameters, from `configs/experiments/ppo_training.yaml` (paper Table 4):

| Parameter | Value |
| :--- | :--- |
| Episodes | 1000 |
| Timesteps per episode | 3000 |
| Grid | 100 × 100 |
| UAV fleet *N* | 20 |
| Sectors *Z* | 25 |
| Learning rate | 3 × 10⁻⁴ |
| Clip ratio | 0.2 |
| Entropy coefficient | 0.01 |
| Discount γ | 0.99 |
| Epochs per update | 4 |
| Hidden dims | [256, 128] |
| Seed | 42 |

### Multi-seed — the manuscript's validation curve

```bash
python experiments/11c_train_multiseed.py --seeds 5 --episodes 1000 --workers 5
```

Writes per-seed checkpoints and learning curves plus an aggregate mean/std
curve. It appends after every episode, so an interrupted run still yields
usable data — unlike `11_ppo_training.py`, which writes only at the end.

### Without local compute

`notebooks/colab_train_ppo_gomdp.ipynb` runs the same training on Google Colab.
Set `--outdir` to a local path, then copy the resulting `best_checkpoint.pt` to
`src/wildfire_governance/rl/checkpoints/ppo_gomdp_best.pt`.

---

## After training

```bash
python -m wildfire_governance.rl.evaluator --n_seeds 20      # evaluate
python experiments/11b_rl_comparison.py                      # regenerate Table 1
```

Expected, matching Table 1:

| Metric | Value |
| :--- | :--- |
| *L_d* (detection latency) | 15.1 ± 1.1 steps |
| *F_p* (false public alerts) | 6.0 ± 1.1 % |
| *FN_r* | 2.1 ± 0.9 % |
| Governance compliance | 100.0 % |

Seed noise means a fresh run will not match to the last decimal. The intended
tolerance is 5 % relative:

```bash
bash scripts/check_reproducibility.sh
```

Governance compliance is the exception: it is **exactly** 100 %, for any policy,
by Theorem 1. That column is enforced by the environment, not learned — an
untrained policy also scores 100 % while its latency and false-alert numbers are
far off. That contrast is a useful sanity check in itself, and you can reproduce
it deliberately:

```bash
python -m wildfire_governance.rl.evaluator --n_seeds 5 --allow-untrained
```

`--allow-untrained` is accepted by every checkpoint-dependent script. It is a
Theorem 1 control, **not** a reproduction of the paper's numbers, and it prints a
banner saying so.

---

## Using a checkpoint from an older revision

Checkpoints trained before the policy head was fused store one `nn.Linear` per
UAV (`heads.0…heads.N`); the current network uses a single fused `head`.
`PPOGOMDPAgent.load_state_dict()` migrates these automatically by concatenating
the per-UAV blocks, which reproduces the fused layer exactly — old checkpoints
load without conversion.

The optimiser state is dropped during migration (its parameter keys no longer
match), so training resumes with a fresh Adam state. Evaluation is unaffected.
