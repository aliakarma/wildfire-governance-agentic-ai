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

FIX (Issue 1): Replaced random-action log-prob computation with a proper
RolloutBuffer that stores (obs, actions_per_uav, log_probs_old, values,
rewards, dones).  update() now uses stored log-probs and implements the
correct clipped surrogate objective: L^CLIP = E[min(r_t*A_t, clip(r_t,
1-eps, 1+eps)*A_t)].
"""
from __future__ import annotations

from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Rollout buffer (FIX Issue 1 — was missing entirely)
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """Stores one episode of (obs, actions, log_probs_old, values, rewards, dones).

    All tensors are accumulated via append() and converted to torch tensors
    in as_tensors().
    """

    observations: List[np.ndarray] = field(default_factory=list)
    # actions_per_uav[t] is a LongTensor of shape (n_uavs,)
    actions_per_uav: List[List[int]] = field(default_factory=list)
    # log_probs_old[t] is a list of per-UAV scalar log-probs
    log_probs_old: List[List[float]] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def append(
        self,
        obs: np.ndarray,
        actions: List[int],
        log_probs: List[float],
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.observations.append(obs)
        self.actions_per_uav.append(actions)
        self.log_probs_old.append(log_probs)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self) -> None:
        self.__init__()  # type: ignore[misc]

    def __len__(self) -> int:
        return len(self.rewards)

    def as_tensors(self, gamma: float, device: str = "cpu") -> Dict[str, "torch.Tensor"]:
        """Convert buffer to dict of torch tensors with computed returns."""
        import torch

        obs_t = torch.FloatTensor(np.array(self.observations)).to(device)
        n_uavs = len(self.actions_per_uav[0])
        T = len(self.rewards)

        # (T, n_uavs)
        act_t = torch.LongTensor(self.actions_per_uav).to(device)
        # (T, n_uavs)
        old_lp_t = torch.FloatTensor(self.log_probs_old).to(device)
        val_t = torch.FloatTensor(self.values).to(device)

        # Discounted returns
        returns = _compute_returns(self.rewards, gamma)
        ret_t = torch.FloatTensor(returns).to(device)

        return {
            "obs": obs_t,
            "actions": act_t,
            "old_log_probs": old_lp_t,
            "values": val_t,
            "returns": ret_t,
        }


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """2-layer MLP policy network for PPO-GOMDP."""

    def __init__(self, obs_dim: int, n_uavs: int, n_sectors: int) -> None:
        _require_torch()
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(128, n_sectors) for _ in range(n_uavs)]
        )

    def forward(self, obs: "torch.Tensor") -> List["torch.Tensor"]:  # type: ignore[name-defined]
        shared_out = self.shared(obs)
        return [head(shared_out) for head in self.heads]


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


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

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
    ) -> None:
        _require_torch()
        self.grid_size = grid_size
        self.n_uavs = n_uavs
        self.n_sectors = n_sectors
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.n_epochs = n_epochs

        obs_dim = grid_size * grid_size + 2 * n_uavs
        self.policy = PolicyNetwork(obs_dim, n_uavs, n_sectors)
        self.value_net = ValueNetwork(obs_dim)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=lr,
        )

        self._training_step: int = 0
        # Rollout buffer — populated during episode collection
        self._buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    # Action selection — stores log_probs for the rollout buffer
    # ------------------------------------------------------------------

    def select_actions(
        self,
        obs: np.ndarray,
        uav_fleet: Any,
    ) -> Tuple[Dict[int, int], List[float], float]:
        """Select sector assignments for all UAVs.

        Returns:
            Tuple of (allocation_dict, per_uav_log_probs, value_estimate).
            allocation_dict maps uav_index → sector_id.
            per_uav_log_probs is a list of scalar log-probs (one per UAV).
            value_estimate is the scalar V(obs).
        """
        import torch
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits_list = self.policy(obs_tensor)
            value = float(self.value_net(obs_tensor).item())

        allocation: Dict[int, int] = {}
        log_probs: List[float] = []
        for i, logits in enumerate(logits_list):
            if i < len(uav_fleet) and hasattr(uav_fleet[i], "battery_fraction"):
                if uav_fleet[i].battery_fraction < 0.05:
                    # Recharging UAV — keep last sector (or 0)
                    allocation[i] = 0
                    log_probs.append(0.0)
                    continue
            dist = Categorical(logits=logits.squeeze(0))
            sector = dist.sample()
            allocation[i] = int(sector.item())
            log_probs.append(float(dist.log_prob(sector).item()))

        return allocation, log_probs, value

    def select_actions_greedy(
        self,
        obs: np.ndarray,
        uav_fleet: Any,
    ) -> Dict[int, int]:
        """Action selection without storing log_probs (inference only)."""
        allocation, _, _ = self.select_actions(obs, uav_fleet)
        return allocation

    # ------------------------------------------------------------------
    # Buffer helpers (called by trainer)
    # ------------------------------------------------------------------

    def store_transition(
        self,
        obs: np.ndarray,
        actions: List[int],
        log_probs: List[float],
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        """Append one transition to the rollout buffer."""
        self._buffer.append(obs, actions, log_probs, value, reward, done)

    def clear_buffer(self) -> None:
        """Clear the rollout buffer at the start of a new episode."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # PPO update — FIX Issue 1: use stored log-probs, clipped surrogate
    # ------------------------------------------------------------------

    def update_from_buffer(self) -> float:
        """Run PPO update using the current rollout buffer.

        Implements:
            L^CLIP = E[min(r_t*A_t, clip(r_t, 1-eps, 1+eps)*A_t)]
        where r_t = pi_new(a|s) / pi_old(a|s).

        Returns:
            Mean total loss across n_epochs.
        """
        import torch

        if len(self._buffer) == 0:
            return 0.0

        batch = self._buffer.as_tensors(self.gamma)
        obs_t = batch["obs"]           # (T, obs_dim)
        act_t = batch["actions"]       # (T, n_uavs)
        old_lp_t = batch["old_log_probs"]  # (T, n_uavs)
        ret_t = batch["returns"]       # (T,)

        # Normalise returns
        if ret_t.std() > 1e-8:
            ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

        total_loss_acc = 0.0
        for _ in range(self.n_epochs):
            values = self.value_net(obs_t)                   # (T,)
            advantages = (ret_t - values.detach())
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            logits_list = self.policy(obs_t)                 # list of (T, Z)
            policy_loss = torch.tensor(0.0)
            entropy_bonus = torch.tensor(0.0)

            for uav_idx, logits in enumerate(logits_list):
                dist = Categorical(logits=logits)            # (T,)
                # NEW log-probs for actions actually taken
                new_lp = dist.log_prob(act_t[:, uav_idx])   # (T,)
                old_lp = old_lp_t[:, uav_idx]               # (T,)

                ratio = torch.exp(new_lp - old_lp)           # r_t
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio,
                ) * advantages
                policy_loss = policy_loss - torch.min(surr1, surr2).mean()
                entropy_bonus = entropy_bonus - dist.entropy().mean()

            value_loss = nn.functional.mse_loss(values, ret_t)
            loss = (
                policy_loss
                + 0.5 * value_loss
                + self.entropy_coeff * entropy_bonus
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            total_loss_acc += loss.item()

        self._training_step += 1
        return total_loss_acc / self.n_epochs

    # Backward-compat wrapper used by old trainer
    def update(
        self,
        observations: List[np.ndarray],
        actions: List[Dict[int, int]],
        rewards: List[float],
        dones: List[bool],
    ) -> float:
        """Legacy update interface (populates buffer then calls update_from_buffer)."""
        self.clear_buffer()
        import torch
        obs_tensor = torch.FloatTensor(np.array(observations))
        with torch.no_grad():
            logits_list = self.policy(obs_tensor)
            values = self.value_net(obs_tensor)
        for t, (obs, act_dict, reward, done) in enumerate(
            zip(observations, actions, rewards, dones)
        ):
            act_list = [act_dict.get(i, 0) for i in range(self.n_uavs)]
            lp_list = []
            for uav_idx, logits in enumerate(logits_list):
                from torch.distributions import Categorical as Cat
                d = Cat(logits=logits[t])
                lp_list.append(float(d.log_prob(torch.tensor(act_list[uav_idx])).item()))
            self._buffer.append(obs, act_list, lp_list, float(values[t].item()), reward, done)
        return self.update_from_buffer()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        """Save policy, value network, and optimiser state."""
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
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
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        """Load from a saved checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        import torch
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"PPO-GOMDP checkpoint not found: {path}\n"
                "Run: make train-ppo  OR  use --use_pretrained with the pre-trained checkpoint."
            )
        ckpt = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.value_net.load_state_dict(ckpt["value_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._training_step = ckpt.get("training_step", 0)


def _compute_returns(rewards: List[float], gamma: float) -> List[float]:
    """Compute discounted cumulative returns."""
    returns: List[float] = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    return returns
