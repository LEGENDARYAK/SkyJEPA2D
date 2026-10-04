"""Waypoint control through a trained model using MPPI candidate rollouts.

Only state history, executed commands, and the waypoint enter the planner.
There is no access to the plant's hidden motors or future true states, and
there is no silent fallback to the conventional waypoint controller.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .config import SimConfig
from .training import load_checkpoint


class LearnedController:
    def __init__(
        self,
        checkpoint,
        config: SimConfig,
        device="cpu",
        samples=128,
        horizon=25,
        iterations=2,
        seed=13,
        zero_residual=False,
    ):
        if samples < 8 or horizon < 2 or iterations < 1:
            raise ValueError("Planner requires samples >= 8, horizon >= 2, iterations >= 1")
        self.model, self.prober, self.metadata, self.device = load_checkpoint(checkpoint, device)
        nominal = self.model.config
        for name in ("control_dt", "physics_dt", "gravity", "width", "height"):
            if not np.isclose(getattr(config.world, name), getattr(nominal.world, name)):
                raise ValueError(f"Checkpoint and simulator have different world.{name}")
        self.config = nominal
        self.history = self.metadata["history"]
        self.samples, self.horizon, self.iterations = samples, horizon, iterations
        self.rng = np.random.default_rng(seed)
        self.zero_residual = zero_residual
        self.hover = np.full(2, nominal.drone.mass * nominal.world.gravity / 2)
        self.reset([0, 0, 0, 0, 0, 0])

    def reset(self, state):
        self.states = deque(
            [np.asarray(state).copy() for _ in range(self.history)], maxlen=self.history
        )
        self.commands = deque(
            [self.hover.copy() for _ in range(self.history - 1)], maxlen=max(1, self.history - 1)
        )
        self.plan = np.tile(self.hover, (self.horizon, 1))

    def observe(self, state, executed_action):
        self.states.append(np.asarray(state).copy())
        if self.history > 1:
            self.commands.append(np.asarray(executed_action).copy())

    def _tensor(self, array):
        return torch.as_tensor(np.asarray(array, dtype=np.float32), device=self.device)

    @torch.no_grad()
    def action(self, state, goal):
        # The latest observation must be available even for callers that reset at launch.
        self.states[-1] = np.asarray(state).copy()
        context = self._tensor(np.array(self.states))[None].expand(self.samples, -1, -1)
        past_array = np.array(self.commands) if self.history > 1 else np.empty((0, 2))
        past = self._tensor(past_array)[None].expand(self.samples, -1, -1)
        initial = self._tensor(state)[None].expand(self.samples, -1)
        target = self._tensor(goal)
        max_force = self.config.drone.max_motor_force
        for _ in range(self.iterations):
            # Smooth collective and differential perturbations excite lift and pitch
            # separately. Candidate zero preserves the current plan for comparison.
            noise = self.rng.normal(size=(self.samples, self.horizon, 2))
            for k in range(1, self.horizon):
                noise[:, k] = 0.82 * noise[:, k - 1] + 0.57 * noise[:, k]
            collective, differential = 0.7 * noise[..., 0], 0.10 * noise[..., 1]
            perturbations = np.stack([collective - differential, collective + differential], -1)
            perturbations[0] = 0
            candidates = np.clip(self.plan[None] + perturbations, 0, max_force)
            actions = self._tensor(candidates)
            latent = self.model.from_actions(context, past, actions)
            predicted = self.prober(
                latent, initial, actions, past, zero_residual=self.zero_residual
            )
            position_error = (predicted[..., :2] - target).square().sum(-1)
            velocity = predicted[..., 2:4].square().sum(-1)
            cost = (
                position_error
                + 0.7 * velocity
                + 2 * predicted[..., 4].square()
                + 0.05 * predicted[..., 5].square()
            ).mean(-1)
            cost = cost + 5 * position_error[:, -1] + velocity[:, -1]
            cost = cost + 0.01 * (actions - self._tensor(self.hover)).square().mean((1, 2))
            w = self.config.world
            r = w.collision_radius + 0.15
            unsafe = (
                (predicted[..., 0] < r)
                | (predicted[..., 0] > w.width - r)
                | (predicted[..., 1] < r)
                | (predicted[..., 1] > w.height - r)
            )
            cost = cost + 10000 * unsafe.float().mean(-1)
            cost = cost + 100 * (predicted[..., 4].abs() - 0.8).clamp_min(0).square().mean(-1)
            finite = torch.isfinite(cost)
            if not finite.any():
                raise FloatingPointError("All learned candidate rollouts are nonfinite")
            cost = torch.where(finite, cost, torch.full_like(cost, float("inf")))
            weights = torch.softmax(-(cost - cost.min()) / 2.0, dim=0)
            self.plan = (weights[:, None, None] * actions).sum(0).cpu().numpy().astype(float)
        command = self.plan[0].copy()
        self.plan[:-1] = self.plan[1:]
        self.plan[-1] = self.hover
        return command
