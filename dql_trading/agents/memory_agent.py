from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

from dql_trading.agents.dql_agent import DQLAgent, QNetwork

class RNNQNetwork(nn.Module):
    """A tiny GRU-based Q-network that keeps temporal context."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(inplace=True)
        )
        self.gru = nn.GRU(128, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(inplace=True), nn.Linear(64, action_dim)
        )
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        """Run a forward pass.

        Accepts both (B, state_dim) and (B, T, state_dim) shapes for *x* so that
        the vanilla DQL training loop (which supplies flat states) still works.
        """
        needs_unsqueeze = False
        if x.dim() == 2:  # (B, state_dim) -> add seq dim
            x = x.unsqueeze(1)
            needs_unsqueeze = True

        z = self.embed(x)  # (B, T, 128)
        out, h_n = self.gru(z, h)  # (B, T, hidden)
        q = self.head(out[:, -1, :])  # last time step

        if h is None:
            # Training mode (no hidden passed in) — return only Q-values.
            return q

        # Inference mode: return both Q-values and updated hidden state.
        return q, h_n


class MemoryAgent(DQLAgent):
    """Deep Recurrent Q-Network agent with a tiny GRU for memory."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-4,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        hidden_size: int = 64,
    ):
        # Initialise base without creating its Q network
        super().__init__(
            state_dim,
            action_dim,
            gamma=gamma,
            lr=lr,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )
        # Replace feed-forward nets with recurrent ones
        self.q_net = RNNQNetwork(state_dim, action_dim, hidden_size)
        self.target_net = RNNQNetwork(state_dim, action_dim, hidden_size)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.update_target()
        self.reset_hidden()

    # ---------------------------------------------------------------------
    # Hidden-state helpers
    # ---------------------------------------------------------------------
    def reset_hidden(self, batch_size: int = 1):
        self.h = torch.zeros(1, batch_size, self.q_net.hidden_size)

    # ---------------------------------------------------------------------
    # Action selection overrides (keeps hidden state)
    # ---------------------------------------------------------------------
    def select_action(self, state, test: bool = False):
        if isinstance(state, np.ndarray) and state.ndim == 1:
            state = state.reshape(1, -1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # (B=1, T=1, dim)
        with torch.no_grad():
            q_values, self.h = self.q_net(state_tensor, self.h.detach())
        if (not test) and (random.random() < self.epsilon):
            return random.randint(0, self.action_dim - 1)
        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        # Ensure 1D storage for compatibility with base training logic
        if isinstance(state, np.ndarray) and state.ndim > 1:
            state = state.flatten()
        if isinstance(next_state, np.ndarray) and next_state.ndim > 1:
            next_state = next_state.flatten()
        self.memory.append((state, action, reward, next_state, done))

    # Training still uses single-step samples (memory not leveraged in loss).
    # For a capstone baseline that's acceptable; it still uses hidden state
    # during inference which provides temporal context. 