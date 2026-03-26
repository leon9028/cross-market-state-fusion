#!/usr/bin/env python3
"""
PyTorch-based PPO (Proximal Policy Optimization) strategy.

Same architecture and hyperparameters as rl_mlx.py, but runs on Linux/Ubuntu/EC2
(CPU or CUDA). Use this when MLX is not available (Apple Silicon only).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from safetensors.torch import save_file as safetensors_save
from safetensors.torch import load_file as safetensors_load
from typing import List, Dict, Optional
from dataclasses import dataclass
from .base import Strategy, MarketState, Action, STATE_FEATURE_DIM


@dataclass
class Experience:
    """Single experience tuple with temporal context."""
    state: np.ndarray
    temporal_state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_temporal_state: np.ndarray
    done: bool
    log_prob: float
    value: float
    cid: str = ""


class TemporalEncoder(nn.Module):
    """Encodes temporal sequence of states into momentum/trend features."""

    def __init__(self, input_dim: int = STATE_FEATURE_DIM, history_len: int = 5, output_dim: int = 32):
        super().__init__()
        self.history_len = history_len
        self.temporal_input = input_dim * history_len
        self.fc1 = nn.Linear(self.temporal_input, 64)
        self.ln1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.ln1(self.fc1(x)))
        h = torch.tanh(self.ln2(self.fc2(h)))
        return h


class Actor(nn.Module):
    """Policy network with temporal awareness."""

    def __init__(self, input_dim: int = STATE_FEATURE_DIM, hidden_size: int = 64, output_dim: int = 3,
                 history_len: int = 5, temporal_dim: int = 32):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(input_dim, history_len, temporal_dim)
        combined_dim = input_dim + temporal_dim
        self.fc1 = nn.Linear(combined_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

    ACTION_FLOOR = 0.05

    def forward(self, current_state: torch.Tensor, temporal_state: torch.Tensor) -> torch.Tensor:
        temporal_features = self.temporal_encoder(temporal_state)
        combined = torch.cat([current_state, temporal_features], dim=-1)
        h = torch.tanh(self.ln1(self.fc1(combined)))
        h = torch.tanh(self.ln2(self.fc2(h)))
        logits = self.fc3(h)
        probs = torch.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=self.ACTION_FLOOR)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs


class Critic(nn.Module):
    """Value network with temporal awareness (larger than actor)."""

    def __init__(self, input_dim: int = STATE_FEATURE_DIM, hidden_size: int = 128,
                 history_len: int = 5, temporal_dim: int = 32):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(input_dim, history_len, temporal_dim)
        combined_dim = input_dim + temporal_dim
        self.fc1 = nn.Linear(combined_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, current_state: torch.Tensor, temporal_state: torch.Tensor) -> torch.Tensor:
        temporal_features = self.temporal_encoder(temporal_state)
        combined = torch.cat([current_state, temporal_features], dim=-1)
        h = torch.tanh(self.ln1(self.fc1(combined)))
        h = torch.tanh(self.ln2(self.fc2(h)))
        value = self.fc3(h)
        return value


class RLStrategy(Strategy):
    """PPO-based strategy with temporal actor-critic using PyTorch (runs on Ubuntu/EC2)."""

    def __init__(
        self,
        input_dim: int = STATE_FEATURE_DIM,
        hidden_size: int = 64,
        critic_hidden_size: int = 192,
        history_len: int = 5,
        temporal_dim: int = 32,
        lr_actor: float = 3e-5,
        lr_critic: float = 2e-4,
        gamma: float = 0.80,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.15,
        entropy_coef: float = 0.07,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        buffer_size: int = 2048,
        batch_size: int = 128,
        n_epochs: int = 3,
        n_critic_extra_epochs: int = 3,
        target_kl: float = 0.02,
    ):
        super().__init__("rl")
        # Low-memory override: set RL_BUFFER_SIZE and/or RL_BATCH_SIZE (e.g. 128, 32) when <2GB RAM
        _buf = os.environ.get("RL_BUFFER_SIZE")
        _bs = os.environ.get("RL_BATCH_SIZE")
        buffer_size = int(_buf) if _buf else buffer_size
        batch_size = int(_bs) if _bs else batch_size

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.history_len = history_len
        self.temporal_dim = temporal_dim
        self.output_dim = 3

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_critic_extra_epochs = n_critic_extra_epochs
        self.target_kl = target_kl

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(input_dim, hidden_size, self.output_dim, history_len, temporal_dim).to(self.device)
        self.critic = Critic(input_dim, critic_hidden_size, history_len, temporal_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.experiences: List[Experience] = []
        self._state_history: Dict[str, deque] = {}

        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._last_temporal_state: Optional[np.ndarray] = None

    def _get_temporal_state(self, asset: str, current_features: np.ndarray) -> np.ndarray:
        if asset not in self._state_history:
            self._state_history[asset] = deque(maxlen=self.history_len)
        history = self._state_history[asset]
        history.append(current_features.copy())
        if len(history) < self.history_len:
            padding = [np.zeros(self.input_dim, dtype=np.float32)] * (self.history_len - len(history))
            stacked = np.concatenate(padding + list(history))
        else:
            stacked = np.concatenate(list(history))
        return stacked.astype(np.float32)

    def act(self, state: MarketState) -> Action:
        features = state.to_features()
        temporal_state = self._get_temporal_state(state.asset, features)

        features_t = torch.from_numpy(features.reshape(1, -1).astype(np.float32)).to(self.device)
        temporal_t = torch.from_numpy(temporal_state.reshape(1, -1).astype(np.float32)).to(self.device)

        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            probs = self.actor(features_t, temporal_t)
            value = self.critic(features_t, temporal_t)

        probs_np = probs.cpu().numpy().flatten()
        value_np = float(value.cpu().item())

        if self.training:
            action_idx = np.random.choice(self.output_dim, p=probs_np)
        else:
            action_idx = int(np.argmax(probs_np))

        self._last_log_prob = float(np.log(probs_np[action_idx] + 1e-8))
        self._last_value = value_np
        self._last_temporal_state = temporal_state

        return Action(action_idx)

    def store(self, state: MarketState, action: Action, reward: float,
              next_state: MarketState, done: bool,
              log_prob: float = None, value: float = None,
              temporal_state: np.ndarray = None,
              cid: str = ""):
        """When log_prob/value/temporal_state are provided explicitly (per-cid),
        they take precedence over self._last_* which may belong to a different cid."""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        self.reward_std = np.sqrt(
            ((self.reward_count - 1) * self.reward_std**2 + delta * (reward - self.reward_mean))
            / max(1, self.reward_count)
        )

        norm_reward = reward / max(1.0, self.reward_std)

        next_features = next_state.to_features()
        next_temporal_state = self._get_temporal_state(next_state.asset, next_features)

        _log_prob = log_prob if log_prob is not None else self._last_log_prob
        _value = value if value is not None else self._last_value
        _temporal = temporal_state if temporal_state is not None else (
            self._last_temporal_state if self._last_temporal_state is not None
            else np.zeros(self.history_len * self.input_dim, dtype=np.float32)
        )

        exp = Experience(
            state=state.to_features(),
            temporal_state=_temporal,
            action=action.value,
            reward=norm_reward,
            next_state=next_features,
            next_temporal_state=next_temporal_state,
            done=done,
            log_prob=_log_prob,
            value=_value,
            cid=cid,
        )
        self.experiences.append(exp)
        if len(self.experiences) > self.buffer_size:
            self.experiences = self.experiences[-self.buffer_size:]

    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                     dones: np.ndarray, next_value: float) -> tuple:
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)
        gae = 0.0
        for t in reversed(range(n)):
            next_val = next_value if t == n - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        return advantages, returns

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.experiences) < self.buffer_size:
            return None

        n = len(self.experiences)
        states = np.array([e.state for e in self.experiences], dtype=np.float32)
        temporal_states = np.array([e.temporal_state for e in self.experiences], dtype=np.float32)
        actions = np.array([e.action for e in self.experiences], dtype=np.int64)
        rewards = np.array([e.reward for e in self.experiences], dtype=np.float32)
        dones = np.array([e.done for e in self.experiences], dtype=np.float32)
        old_log_probs = np.array([e.log_prob for e in self.experiences], dtype=np.float32)
        old_values = np.array([e.value for e in self.experiences], dtype=np.float32)

        # Per-market GAE: group by cid to avoid cross-market TD leakage
        from collections import defaultdict
        market_indices = defaultdict(list)
        for i, exp in enumerate(self.experiences):
            market_indices[exp.cid].append(i)

        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        with torch.no_grad():
            for cid, indices in market_indices.items():
                m_rewards = rewards[indices]
                m_values = old_values[indices]
                m_dones = dones[indices]

                last_exp = self.experiences[indices[-1]]
                if last_exp.done:
                    next_value = 0.0
                else:
                    ns = torch.from_numpy(last_exp.next_state.reshape(1, -1)).to(self.device)
                    nt = torch.from_numpy(last_exp.next_temporal_state.reshape(1, -1)).to(self.device)
                    next_value = float(self.critic(ns, nt).cpu().item())

                m_adv, m_ret = self._compute_gae(m_rewards, m_values, m_dones, next_value)
                for j, idx in enumerate(indices):
                    advantages[idx] = m_adv[j]
                    returns[idx] = m_ret[j]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.from_numpy(states).to(self.device)
        temporal_t = torch.from_numpy(temporal_states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        old_log_probs_t = torch.from_numpy(old_log_probs).to(self.device)
        advantages_t = torch.from_numpy(advantages.astype(np.float32)).to(self.device)
        returns_t = torch.from_numpy(returns.astype(np.float32)).to(self.device)
        old_values_t = torch.from_numpy(old_values).to(self.device)

        n_samples = len(self.experiences)
        all_metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []}

        self.actor.train()
        self.critic.train()

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            epoch_kl = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                idx = indices[start:end]

                batch_states = states_t[idx]
                batch_temporal = temporal_t[idx]
                batch_actions = actions_t[idx]
                batch_old_log_probs = old_log_probs_t[idx]
                batch_advantages = advantages_t[idx]
                batch_returns = returns_t[idx]
                batch_old_values = old_values_t[idx]

                # Actor loss
                probs = self.actor(batch_states, batch_temporal)
                log_probs = torch.log(probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1) + 1e-8)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                policy_loss = policy_loss - self.entropy_coef * entropy

                approx_kl = (batch_old_log_probs - log_probs).mean().item()
                clip_frac = ((ratio < 1 - self.clip_epsilon) | (ratio > 1 + self.clip_epsilon)).float().mean().item()

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic loss — Smooth L1 (Huber) vs MSE: less blow-up on PnL spikes
                values = self.critic(batch_states, batch_temporal).squeeze(-1)
                value_loss = F.smooth_l1_loss(values, batch_returns, beta=1.0, reduction="mean")

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                all_metrics["policy_loss"].append(policy_loss.item())
                all_metrics["value_loss"].append(value_loss.item())
                all_metrics["entropy"].append(entropy.item())
                all_metrics["approx_kl"].append(approx_kl)
                all_metrics["clip_fraction"].append(clip_frac)
                epoch_kl += approx_kl
                n_batches += 1

            avg_kl = epoch_kl / max(n_batches, 1)
            if avg_kl > self.target_kl:
                print(f"  [RL] Early stop epoch {epoch}, KL={avg_kl:.4f}")
                break

        # Extra critic-only epochs (actor frozen) to lift explained_variance
        for _ in range(self.n_critic_extra_epochs):
            c_indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                idx = c_indices[start:end]
                b_states = states_t[idx]
                b_temporal = temporal_t[idx]
                b_returns = returns_t[idx]

                v = self.critic(b_states, b_temporal).squeeze(-1)
                c_loss = F.smooth_l1_loss(v, b_returns, beta=1.0, reduction="mean")
                self.critic_optimizer.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

        # Compute diagnostics before clearing buffer
        var_y = np.var(returns)
        explained_var = 1 - np.var(returns - old_values) / (var_y + 1e-8) if var_y > 0 else 0.0

        raw_adv = returns - old_values
        avg_advantage = float(np.mean(raw_adv))
        advantage_std = float(np.std(raw_adv))

        avg_value = float(np.mean(old_values))
        value_range = float(np.max(old_values) - np.min(old_values))

        action_counts = [0, 0, 0]
        for a in actions:
            action_counts[int(a)] += 1
        total_actions = max(1, sum(action_counts))
        hold_pct = action_counts[0] / total_actions
        buy_pct = action_counts[1] / total_actions
        sell_pct = action_counts[2] / total_actions

        self.experiences.clear()

        return {
            "policy_loss": np.mean(all_metrics["policy_loss"]),
            "value_loss": np.mean(all_metrics["value_loss"]),
            "entropy": np.mean(all_metrics["entropy"]),
            "approx_kl": np.mean(all_metrics["approx_kl"]),
            "clip_fraction": np.mean(all_metrics["clip_fraction"]),
            "explained_variance": explained_var,
            "avg_value": avg_value,
            "value_range": value_range,
            "avg_advantage": avg_advantage,
            "advantage_std": advantage_std,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "hold_pct": hold_pct,
            "buy_pct": buy_pct,
            "sell_pct": sell_pct,
        }

    def reset(self):
        self.experiences.clear()
        self._state_history.clear()
        self._last_temporal_state = None

    def save(self, path: str):
        """Save in same format as rl_mlx: .safetensors (weights) + _stats.npz (stats)."""
        base = path.replace(".npz", "").replace(".safetensors", "").replace(".pt", "")
        # Flatten keys with actor./critic. prefix to match rl_mlx format
        weights = {}
        for k, v in self.actor.state_dict().items():
            weights["actor." + k] = v
        for k, v in self.critic.state_dict().items():
            weights["critic." + k] = v
        safetensors_save(weights, base + ".safetensors")
        np.savez(
            base + "_stats.npz",
            reward_mean=self.reward_mean,
            reward_std=self.reward_std,
            reward_count=self.reward_count,
            input_dim=self.input_dim,
            hidden_size=self.hidden_size,
            critic_hidden_size=self.critic_hidden_size,
            history_len=self.history_len,
            temporal_dim=self.temporal_dim,
            gamma=self.gamma,
            buffer_size=self.buffer_size,
        )

    def load(self, path: str):
        """Load from .safetensors + _stats.npz (same as rl_mlx). Falls back to .pt if no safetensors."""
        base = path.replace(".npz", "").replace(".safetensors", "").replace(".pt", "")
        try:
            weights = safetensors_load(base + ".safetensors", device=str(self.device))
            actor_sd = {k.replace("actor.", "", 1): v for k, v in weights.items() if k.startswith("actor.")}
            critic_sd = {k.replace("critic.", "", 1): v for k, v in weights.items() if k.startswith("critic.")}
            self.actor.load_state_dict(actor_sd, strict=True)
            self.critic.load_state_dict(critic_sd, strict=True)
        except FileNotFoundError:
            ckpt = torch.load(base + ".pt", map_location=self.device, weights_only=True)
            self.actor.load_state_dict(ckpt["actor"])
            self.critic.load_state_dict(ckpt["critic"])
        stats = np.load(base + "_stats.npz", allow_pickle=True)
        self.reward_mean = float(stats["reward_mean"])
        self.reward_std = float(stats["reward_std"])
        self.reward_count = int(stats["reward_count"])
        self.actor.eval()
        self.critic.eval()
