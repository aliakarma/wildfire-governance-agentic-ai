# Paper Result CSVs — Caveats and Metric Definitions

This directory contains the pre-committed result CSVs used to generate paper figures and tables.

## Metric Definitions

### False Positive Rate (Fp)
- **Definition**: False Discovery Rate (FDR) = (# false alerts) / (# total alerts broadcast) × 100%
- **NOT**: Classical false positive rate in signal processing
- **Paper location**: Table II, Section VI-B
- **Computation**: [experiments/utils/runner.py](../../experiments/utils/runner.py), line ~356

### Detection Latency (Ld)
- **Definition**: Timesteps from **actual ignition time** to first detection (confidence > 0.60)
- **Note**: Measured from environment-provided ignition_time (queried per-step), not assumed to be t=0
- **Paper location**: Section VI-B, Equation (Definition 1)

### Governance Compliance
- **Definition**: Fraction of broadcast alerts that carry valid cryptographic governance certificates
- **GOMDP configs**: 100% by Theorem 1 (environment-level enforcement)
- **CMDP configs**: ~92–93% (no blockchain enforcement, HITL approval alone)

## CSV-Specific Notes

### table4_ablation.csv
- **Injections_blocked / Injections_total**: Deterministic *per configuration*
  - With blockchain (`ppo_gomdp_full`, `greedy_gomdp_full`, etc.): 100/100 (smart contract blocks all)
  - Without blockchain (`minus_blockchain`, `ppo_cmdp`): 0/100 (no enforcement, all attempts succeed)
- This reflects the binary enforcement logic (either blockchain is active or it is not) rather than probabilistic simulation.
- For adversarial stress tests with stochastic Byzantine behavior, see `table5_adversarial.csv`.

### table2_rl_comparison.csv
- PPO-CMDP row: Results shown are for **untrained PPO-CMDP** (use_pretrained=False in experiments/11b_rl_comparison.py)
  - This is equivalent to a **Random-CMDP** baseline for fair comparison against trained PPO-GOMDP
  - For a trained PPO-CMDP agent, rerun with `--use_pretrained` flag after training a CMDP checkpoint

## Configuration Caveats

### base.yaml
- **governance.hitl.rejection_rate: 0.15**: Human operator rejects 15% of alerts
  - Non-zero rejection adds filtering benefit beyond latency
  - Used for all experiments (default config)
- **blockchain.consensus.mean_delay_steps: 1.2**: Average consensus delay in steps
  - This is a simulated consensus model (Python); not real Hyperledger Fabric deployment
  - Proposition 1 latency bound: E[Ld] ≤ A/(v·N) + delta, where delta ≈ 1.2 + 3.0 (BC + HV)

## See Also
- [docs/blockchain_setup.md](../../docs/blockchain_setup.md) — Blockchain simulation parameters
- [experiments/README](../../experiments/README.md) — Experiment script guide
