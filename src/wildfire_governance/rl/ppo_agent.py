"""PPO-GOMDP deep reinforcement learning agent.

Trains a Proximal Policy Optimisation (PPO) agent within the GOMDP environment.
The governance constraint is NOT in the reward — it is enforced transparently by
the GovernanceInvariantMDP environment (Theorem 1: Policy-Agnostic Safety).
The agent learns to coordinate UAVs efficiently; governance is guaranteed by the
environment regardless of the policy's training progress.

Architecture: 2-layer MLP (256 → 128 hidden dims, ReLU activations).
Input: flattened belief map (100×100) + UAV positions (2×N floats).
Output: sector assignment logits for each UAV (N × Z logits).

Training hyperparameters (paper Table I):
    lr=3e-4, clip_ratio=0.2, entropy_coeff=0.01, gamma=0.99, n_epochs=4
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for PPO-GOMDP. "
            "Install it with: pip install torch==2.2.1"
        )


class PolicyNetwork(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """2-layer MLP policy network for PPO-GOMDP.

    Args:
        obs_dim: Input dimension (grid_size^2 + 2*n_uavs).
        n_uavs: Number of UAVs (each gets Z sector logits).
        n_sectors: Number of patrol sectors Z.
    """

    def __init__(self, obs_dim: int, n_uavs: int, n_sectors: int) -> None:
        _require_torch()
        super().__init__()
        self.n_uavs = n_uavs
        self.n_sectors = n_sectors
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # One fused head rather than n_uavs small ones. Mathematically identical
        # to independent per-UAV heads (the weight block for UAV i sees only its
        # own output slice), but it issues a single matmul instead of n_uavs
        # kernel launches -- which dominates wall-clock during batch-1 rollout
        # on an accelerator.
        self.head = nn.Linear(128, n_uavs * n_sectors)

    def forward(self, obs: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        """Return per-UAV logits shaped ``[..., n_uavs, n_sectors]``."""
        shared_out = self.shared(obs)
        logits = self.head(shared_out)
        return logits.view(*logits.shape[:-1], self.n_uavs, self.n_sectors)


class ValueNetwork(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Separate value network with same architecture as policy."""

    def __init__(self, obs_dim: int) -> None:
        _require_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        return self.net(obs).squeeze(-1)


class PPOGOMDPAgent:
    """PPO agent trained inside the GOMDP environment.

    Key design: governance constraint is NOT in the reward.
    The GovernanceInvariantMDP environment blocks non-compliant alert
    actions transparently, enforcing Theorem 1 for this agent by construction.

    Args:
        grid_size: Environment grid side length (default 100).
        n_uavs: Number of UAV agents (default 20).
        n_sectors: Number of patrol sectors (default 25).
        lr: Learning rate (paper default: 3e-4).
        clip_ratio: PPO clipping ratio epsilon (paper default: 0.2).
        entropy_coeff: Entropy bonus coefficient (paper default: 0.01).
        gamma: Discount factor (paper default: 0.99).
        n_epochs: PPO update epochs per episode (paper default: 4).
    """

    def __init__(
        self,
        grid_size: int = 100,
        n_uavs: int = 20,
        n_sectors: int = 25,
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        gamma: float = 0.99,
        n_epochs: int = 4,
        device: "str | None" = None,
    ) -> None:
        _require_torch()
        import torch

        self.grid_size = grid_size
        self.n_uavs = n_uavs
        self.n_sectors = n_sectors
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.n_epochs = n_epochs
        # Rollout is dominated by per-step policy forward passes (~72% of
        # episode wall-clock on CPU), so honouring an available accelerator
        # matters more here than for the update step.
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        obs_dim = grid_size * grid_size + 2 * n_uavs
        self.policy = PolicyNetwork(obs_dim, n_uavs, n_sectors).to(self.device)
        self.value_net = ValueNetwork(obs_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=lr,
        )

        self._episode_rewards: List[float] = []
        self._episode_lds: List[float] = []
        self._training_step: int = 0

    def set_learning_rate(self, lr: float) -> None:
        """Set the optimiser learning rate, for schedules driven by the caller.

        Args:
            lr: New learning rate, applied to every parameter group.
        """
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def state_dict(self) -> Dict[str, Any]:
        """Return serializable training state for checkpointing."""
        return {
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self._training_step,
            "config": {
                "grid_size": self.grid_size,
                "n_uavs": self.n_uavs,
                "n_sectors": self.n_sectors,
                "gamma": self.gamma,
                "clip_ratio": self.clip_ratio,
            },
        }

    @staticmethod
    def _migrate_legacy_heads(policy_sd: Dict[str, Any]) -> Dict[str, Any]:
        """Convert pre-fusion ``heads.<i>.*`` checkpoints to the fused ``head.*``.

        Earlier revisions used one ``nn.Linear`` per UAV. Concatenating those
        blocks along the output axis reproduces the fused layer exactly, so old
        checkpoints stay loadable.
        """
        import torch

        head_keys = sorted(
            (k for k in policy_sd if k.startswith("heads.") and k.endswith(".weight")),
            key=lambda k: int(k.split(".")[1]),
        )
        if not head_keys:
            return policy_sd

        migrated = {k: v for k, v in policy_sd.items() if not k.startswith("heads.")}
        idxs = [int(k.split(".")[1]) for k in head_keys]
        migrated["head.weight"] = torch.cat(
            [policy_sd[f"heads.{i}.weight"] for i in idxs], dim=0
        )
        migrated["head.bias"] = torch.cat(
            [policy_sd[f"heads.{i}.bias"] for i in idxs], dim=0
        )
        return migrated

    def load_state_dict(self, checkpoint: Dict[str, Any]) -> None:
        """Load serializable training state from checkpoint payload."""
        self.policy.load_state_dict(
            self._migrate_legacy_heads(checkpoint["policy_state_dict"])
        )
        self.value_net.load_state_dict(checkpoint["value_state_dict"])
        # A migrated checkpoint's optimiser state keys no longer match the fused
        # parameter set, so skip it rather than fail; training resumes with a
        # fresh Adam state.
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (ValueError, KeyError):
            pass
        self._training_step = checkpoint.get("training_step", 0)

    def select_actions(
        self,
        obs: np.ndarray,
        uav_fleet: Any,
    ) -> Dict[int, int]:
        """Select sector assignments for all UAVs.

        Args:
            obs: Observation vector (flattened belief map + UAV positions).
            uav_fleet: List of UAVAgent objects (used to check battery).

        Returns:
            Dict mapping uav_index → sector_id.
        """
        import torch
        obs_tensor = torch.as_tensor(
            np.asarray(obs), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(obs_tensor)            # [1, n_uavs, n_sectors]
            # Sample every UAV in one batched draw and transfer once. Sampling
            # per head with .item() forces one device sync per UAV per step
            # (n_uavs * T syncs per episode), which is the dominant cost of
            # accelerated rollout; this reduces it to a single sync per step.
            sampled = Categorical(logits=logits).sample()   # [1, n_uavs]
            sectors = sampled.squeeze(0).cpu().numpy()

        allocation: Dict[int, int] = {}
        for i in range(self.n_uavs):
            if i < len(uav_fleet) and hasattr(uav_fleet[i], "battery_fraction"):
                if uav_fleet[i].battery_fraction < 0.05:
                    continue
            allocation[i] = int(sectors[i])
        return allocation

    def update(
        self,
        observations: List[np.ndarray],
        actions: List[Dict[int, int]],
        rewards: List[float],
        dones: List[bool],
    ) -> float:
        """Run PPO update on a collected episode trajectory.

        Args:
            observations: List of observation vectors per timestep.
            actions: List of allocation dicts per timestep.
            rewards: List of reward scalars per timestep.
            dones: List of done flags per timestep.

        Returns:
            Mean policy loss for this update.
        """
        import torch

        if not observations:
            return 0.0

        obs_tensor: torch.Tensor = torch.tensor(
            np.asarray(observations), dtype=torch.float32, device=self.device
        )

        # Build a dense [T, n_uavs] tensor from the actual rollout actions.
        action_tensor: torch.Tensor = torch.zeros(
            (len(actions), self.n_uavs), dtype=torch.long, device=self.device
        )
        for t, action_dict in enumerate(actions):
            for uav_idx, sector_idx in action_dict.items():
                if 0 <= uav_idx < self.n_uavs:
                    action_tensor[t, uav_idx] = int(sector_idx)

        # Compute discounted returns with episode termination support.
        returns: List[float] = []
        running_return = 0.0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                running_return = 0.0
            running_return = float(reward) + self.gamma * running_return
            returns.insert(0, running_return)

        returns_tensor: torch.Tensor = torch.tensor(
            returns, dtype=torch.float32, device=self.device
        )
        if returns_tensor.std() > 1e-8:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )

        # Old log-probabilities are computed from the actual rollout actions.
        # The policy has not been updated yet, so these match the rollout policy.
        with torch.no_grad():
            # [T, n_uavs, n_sectors] -> per-UAV log-probs summed over UAVs,
            # equivalent to accumulating one head at a time but in one op.
            old_logits = self.policy(obs_tensor)
            old_log_probs: torch.Tensor = (
                Categorical(logits=old_logits).log_prob(action_tensor).sum(dim=-1)
            )

        total_loss = 0.0
        clip_eps = 0.2

        for _ in range(self.n_epochs):
            values = self.value_net(obs_tensor)
            advantages = returns_tensor - values.detach()
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            logits = self.policy(obs_tensor)
            dist = Categorical(logits=logits)
            new_log_probs: torch.Tensor = dist.log_prob(action_tensor).sum(dim=-1)
            # Mean over both time and UAV axes, matching the previous
            # mean-over-heads-of-mean-over-time.
            entropy_mean = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            surrogate_1 = ratio * advantages
            surrogate_2 = clipped_ratio * advantages
            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
            value_loss = nn.functional.mse_loss(values, returns_tensor)
            entropy_loss = -entropy_mean

            loss = policy_loss + 0.5 * value_loss + self.entropy_coeff * entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            total_loss += float(loss.item())

        self._training_step += 1
        return total_loss / self.n_epochs

    def save_checkpoint(self, path: str) -> None:
        """Save policy, value network, and optimiser state.

        Args:
            path: Output .pt file path.
        """
        import torch
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path_obj)

    def load_checkpoint(self, path: str) -> None:
        """Load from a saved checkpoint.

        Args:
            path: Path to a .pt checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        import torch
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(
                f"PPO-GOMDP checkpoint not found: {path_obj}\n"
                "Run: make train-ppo  OR  use --use_pretrained with the pre-trained checkpoint."
            )
        self.load_state_dict(torch.load(path_obj, map_location=self.device))


def _compute_returns(rewards: List[float], gamma: float) -> List[float]:
    """Compute discounted cumulative returns.

    Args:
        rewards: Per-timestep rewards.
        gamma: Discount factor.

    Returns:
        List of discounted returns, same length as rewards.
    """
    returns: List[float] = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    return returns
