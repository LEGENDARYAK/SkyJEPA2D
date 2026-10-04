"""Conventional waypoint tracking and attitude-assisted manual control."""

from __future__ import annotations

import numpy as np

from .config import SimConfig
from .physics import wrap_angle


def mix_forces(total: float, torque: float, config: SimConfig) -> np.ndarray:
    """Allocate a collective force and pitch torque to the two motor groups."""
    d = config.drone
    total = np.clip(total, 0, 2 * d.max_motor_force)
    half_delta = torque / (2 * d.arm)
    available = min(total / 2, d.max_motor_force - total / 2)
    half_delta = np.clip(half_delta, -available, available)
    return np.array([total / 2 - half_delta, total / 2 + half_delta])


class WaypointController:
    """Cascaded position/velocity/attitude PD controller; also a data collector.

    This is an explicitly named baseline, not a learned JEPA controller.
    """

    def __init__(self, config: SimConfig, max_speed: float = 3.0):
        self.config = config
        self.max_speed = max_speed

    def action(self, state, goal) -> np.ndarray:
        d, w = self.config.drone, self.config.world
        state = np.asarray(state)
        delta = np.asarray(goal) - state[:2]
        desired_velocity = 1.3 * delta
        speed = np.linalg.norm(desired_velocity)
        if speed > self.max_speed:
            desired_velocity *= self.max_speed / speed
        acceleration = np.clip(2.2 * (desired_velocity - state[2:4]), [-4, -4], [4, 4])
        theta_target = np.clip(
            np.arctan2(-acceleration[0], w.gravity + acceleration[1]), -0.55, 0.55
        )
        angular_acceleration = 65 * wrap_angle(theta_target - state[4]) - 13 * state[5]
        torque = d.inertia * angular_acceleration + d.angular_drag * state[5]
        collective = d.mass * (w.gravity + acceleration[1]) / max(np.cos(state[4]), 0.5)
        return mix_forces(collective, torque, self.config)


class ManualController:
    """Arrow controls command lean and collective thrust; position is never set.

    steer: -1 left, +1 right. lift: -1 descend, +1 ascend.
    Releasing steer requests level attitude. Releasing lift supplies hover trim;
    this is not an altitude-hold or waypoint controller.
    """

    def __init__(self, config: SimConfig):
        self.config = config

    def action(self, state, steer: float, lift: float, cut_motors: bool = False):
        if cut_motors:
            return np.zeros(2)
        d, w = self.config.drone, self.config.world
        theta_target = -0.45 * np.clip(steer, -1, 1)
        torque = d.inertia * (65 * wrap_angle(theta_target - state[4]) - 13 * state[5])
        collective = d.mass * w.gravity * (1 + 0.7 * np.clip(lift, -1, 1))
        return mix_forces(collective, torque, self.config)
