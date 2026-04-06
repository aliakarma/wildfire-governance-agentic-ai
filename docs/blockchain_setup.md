# Blockchain Setup

## Simulation Mode (Default)

The repository simulates Hyperledger Fabric blockchain behaviour in Python.
No actual blockchain installation is needed.

The simulation faithfully implements:
- PBFT-variant consensus with k=7 validators, f=2 Byzantine tolerance
- Ed25519 cryptographic signatures via the `cryptography` Python library
- SHA-3 transaction hashing
- Immutable hash-chain audit log

Configuration: `configs/blockchain/fabric_local.yaml`

## Why Hyperledger Fabric?

From the paper (Section III-A): Hyperledger Fabric was chosen because it:
1. Provides permissioned (not public) blockchain — appropriate for institutional governance
2. Supports Byzantine fault-tolerant PBFT-variant consensus
3. Enables deterministic smart contract execution (chaincode)
4. Has production deployment precedent in enterprise safety systems

## Blockchain Parameters (Paper Values)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Validators (k) | 7 | Standard PBFT deployment |
| Byzantine tolerance (f) | 2 | f ≤ (k−1)/3 = 2 |
| Quorum size | 5 = 2f+1 | PBFT requirement |
| Nominal confirmation delay | 1.2 steps | Locally deployed network |
| Burst delay multiplier | 1.35× | 5× anomaly frequency |
| Hash algorithm | SHA-3 256-bit | NIST standard |
| Signature algorithm | Ed25519 | High performance, 128-bit security |

## Smart Contract Logic

The governance smart contract (Eq. 9 in paper) implements:

```
Alert_t ← 1  iff  Conf^(2)_t > τ  AND  σ_validator is valid (Ed25519)
```

Source: `src/wildfire_governance/blockchain/smart_contract.py`

This is the T_G transition function from Definition 1 (GOMDP), which makes
Theorem 1 (Policy-Agnostic Safety) hold for any policy.
