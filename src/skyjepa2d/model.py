"""A compact planar adaptation of SkyJEPA, not an official reproduction.

Stage 1: shared temporal encoders, action-conditioned latent GRU rollout,
multi-step embedding consistency, and projected Gaussian regularization.
Stage 2: frozen latent model with a differentiable physical residual prober.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import SimConfig


class CausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=dilation)

    def forward(self, x):
        return F.silu(self.conv(F.pad(x, (self.padding, 0))))


class TemporalEncoder(nn.Module):
    def __init__(self, inputs, channels, output):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                CausalConv(a, b, 2**i)
                for i, (a, b) in enumerate(zip((inputs, *channels[:-1]), channels, strict=True))
            ]
        )
        self.projection = nn.Linear(channels[-1], output)

    def forward(self, sequence):
        return self.projection(self.blocks(sequence.transpose(1, 2))[:, :, -1])


class LatentDynamics(nn.Module):
    def __init__(self, config: SimConfig, latent_dim=24):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.state_encoder = TemporalEncoder(7, (8, 8, 16), latent_dim)
        self.action_encoder = TemporalEncoder(2, (4, 4, 8), 8)
        self.predictor = nn.GRUCell(8, latent_dim)

    def state_features(self, state):
        w = self.config.world
        return torch.stack(
            [
                state[..., 0] / (w.width / 2) - 1,
                state[..., 1] / (w.height / 2) - 1,
                state[..., 2] / 3,
                state[..., 3] / 3,
                torch.sin(state[..., 4]),
                torch.cos(state[..., 4]),
                state[..., 5] / 3,
            ],
            dim=-1,
        )

    def encode_states(self, history):
        return self.state_encoder(self.state_features(history))

    def encode_actions(self, history):
        hover = self.config.drone.mass * self.config.world.gravity / 2
        return self.action_encoder((history - hover) / self.config.drone.max_motor_force)

    def forward(self, context, action_windows):
        batch, horizon, history, _ = action_windows.shape
        z = self.encode_actions(action_windows.reshape(batch * horizon, history, 2))
        z = z.reshape(batch, horizon, -1)
        latent = self.encode_states(context)
        predicted = []
        for k in range(horizon):
            latent = self.predictor(z[:, k], latent)
            predicted.append(latent)
        return torch.stack(predicted, dim=1)

    def targets(self, target_contexts):
        batch, horizon, history, _ = target_contexts.shape
        return self.encode_states(target_contexts.reshape(batch * horizon, history, 6)).reshape(
            batch, horizon, self.latent_dim
        )

    def from_actions(self, context, past_actions, future_actions):
        history = context.shape[1]
        commands = torch.cat([past_actions, future_actions], dim=1)
        windows = torch.stack(
            [commands[:, k : k + history] for k in range(future_actions.shape[1])], dim=1
        )
        return self(context, windows)


def sigreg(latents, projections=32, knots=17, seed=None):
    """Sketched Epps-Pulley statistic using real-valued characteristic functions.

    For each rollout time, compare random 1D projections across the batch with
    N(0,1). Integrate on [-3,3] by symmetry and trapezoidal quadrature. The batch
    factor is the conventional finite-sample Epps-Pulley statistic scaling;
    lambda therefore depends on this explicitly documented normalization.
    No complex tensors are needed, including on Apple's MPS backend.
    """
    batch, _, dimension = latents.shape
    if batch < 2:
        raise ValueError("SIGReg requires at least two independent windows per batch")
    if seed is None:
        directions = torch.randn(dimension, projections, device=latents.device, dtype=latents.dtype)
    else:
        generator = torch.Generator().manual_seed(seed)
        directions = torch.randn(
            dimension, projections, generator=generator, dtype=latents.dtype
        ).to(latents.device)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
    projected = latents @ directions  # [batch, time, direction]
    t = torch.linspace(0, 3, knots, device=latents.device, dtype=latents.dtype)
    phase = projected.unsqueeze(-1) * t
    expected = torch.exp(-0.5 * t.square())
    mismatch = (phase.cos().mean(0) - expected).square() + phase.sin().mean(0).square()
    integral = 2 * torch.trapezoid(mismatch * expected, t, dim=-1)
    return batch * integral.mean()


class PhysicalProber(nn.Module):
    """Latent-conditioned residual accelerations around a nominal planar model.

    Deliberate adaptation: retain nominal rotational mechanics and motor lag,
    then learn angular residual K(s) @ a, rather than learn all angular motion.
    A zero residual is an independently measurable physics-only ablation.
    """

    def __init__(self, config: SimConfig, latent_dim=24):
        super().__init__()
        self.config = config
        self.network = nn.Sequential(nn.Linear(latent_dim, 32), nn.SiLU(), nn.Linear(32, 4))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def estimate_motors(self, past_actions, batch):
        d, w = self.config.drone, self.config.world
        motors = past_actions.new_full((batch, 2), d.mass * w.gravity / 2)
        decay = math.exp(-w.control_dt / d.motor_tau) if d.motor_tau else 0.0
        gains = past_actions.new_tensor(d.thrust_gains)
        for k in range(past_actions.shape[1]):
            motors = decay * motors + (1 - decay) * past_actions[:, k] * gains
        return motors

    def forward(self, latents, initial_state, actions, past_actions, zero_residual=False):
        d, w = self.config.drone, self.config.world
        batch, horizon, _ = latents.shape
        residuals = torch.tanh(self.network(latents))
        if zero_residual:
            residuals = torch.zeros_like(residuals)
        motors = self.estimate_motors(past_actions, batch)
        gains = actions.new_tensor(d.thrust_gains)
        wind = actions.new_tensor(d.wind_velocity)
        state = initial_state
        trajectory = []
        h = w.physics_dt
        decay = math.exp(-h / d.motor_tau) if d.motor_tau else 0.0
        for k in range(horizon):
            delta_acc = 3 * residuals[:, k, :2]
            delta_alpha = 0.4 * (residuals[:, k, 2:] * actions[:, k]).sum(dim=-1)
            command = actions[:, k].clamp(0, d.max_motor_force) * gains
            for _ in range(round(w.control_dt / h)):
                next_motors = decay * motors + (1 - decay) * command
                mean_motors = (motors + next_motors) / 2 if d.motor_tau else next_motors
                theta_mid = state[:, 4] + 0.5 * h * state[:, 5]
                total = mean_motors.sum(dim=-1) / d.mass
                acc = torch.stack(
                    [-total * theta_mid.sin(), total * theta_mid.cos() - w.gravity], dim=-1
                )
                acc = acc - d.linear_drag * (state[:, 2:4] - wind) / d.mass + delta_acc
                alpha = (
                    d.arm * (mean_motors[:, 1] - mean_motors[:, 0]) - d.angular_drag * state[:, 5]
                ) / d.inertia + delta_alpha
                position = state[:, :2] + h * state[:, 2:4] + 0.5 * h**2 * acc
                velocity = state[:, 2:4] + h * acc
                theta = state[:, 4] + h * state[:, 5] + 0.5 * h**2 * alpha
                theta = torch.atan2(theta.sin(), theta.cos())
                omega = state[:, 5] + h * alpha
                state = torch.cat([position, velocity, theta[:, None], omega[:, None]], dim=-1)
                motors = next_motors
            trajectory.append(state)
        return torch.stack(trajectory, dim=1)


def physical_loss(predicted, target):
    delta = predicted - target
    angle = torch.atan2(delta[..., 4].sin(), delta[..., 4].cos())
    # Units are normalized explicitly; angle error respects periodicity.
    errors = torch.cat([delta[..., :4], angle.unsqueeze(-1), delta[..., 5:]], dim=-1)
    scale = errors.new_tensor([1, 1, 2, 2, 0.5, 2])
    return (errors / scale).square().mean()


def choose_device(request="auto"):
    if request == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if request == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or auto")
    if request == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu or auto")
    return torch.device(request)
