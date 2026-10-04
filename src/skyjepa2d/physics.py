"""Planar rigid body with two motor groups, integrated with fixed-step RK4.

State: [x, z, vx, vz, theta, omega]. Positive theta rotates the body
counterclockwise in the x/z plane, so positive tilt points thrust to the left.
Actions: [left, right] commanded motor-group forces in newtons.
Actual motor forces are hidden actuator state, recovered indirectly through history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import SimConfig

Array = NDArray[np.float64]
STATE_NAMES = ("x", "z", "vx", "vz", "theta", "omega")
ACTION_NAMES = ("left_force_command", "right_force_command")


def wrap_angle(theta):
    """Map radians to [-pi, pi), for scalars or arrays."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class StepResult:
    state: Array
    motor_forces: Array
    action: Array
    time: float
    terminated: bool
    truncated: bool
    reason: str


class PlanarDrone:
    """Headless plant. Neither keyboard events nor a controller belong here."""

    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()
        self._y = np.zeros(8, dtype=np.float64)
        self.reset()

    @property
    def state(self) -> Array:
        return self._y[:6].copy()

    @property
    def motor_forces(self) -> Array:
        return self._y[6:].copy()

    @property
    def hover_action(self) -> Array:
        d, w = self.config.drone, self.config.world
        return np.full(2, d.mass * w.gravity / 2) / np.asarray(d.thrust_gains)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def reset(self, state=None, motor_forces=None) -> Array:
        w = self.config.world
        initial = [0.2 * w.width, w.height / 3, 0, 0, 0, 0] if state is None else state
        initial = np.asarray(initial, dtype=np.float64)
        if initial.shape != (6,) or not np.isfinite(initial).all():
            raise ValueError("Initial state must contain six finite values")
        r = w.collision_radius
        if not (r < initial[0] < w.width - r and r < initial[1] < w.height - r):
            raise ValueError("Initial position must be inside the flight area")
        motors = (
            np.full(2, self.config.drone.mass * w.gravity / 2)
            if motor_forces is None
            else np.asarray(motor_forces, dtype=np.float64)
        )
        if motors.shape != (2,) or not np.isfinite(motors).all() or (motors < 0).any():
            raise ValueError("Initial motor forces must contain two finite nonnegative values")
        self._y = np.concatenate([initial, motors])
        self._y[4] = wrap_angle(self._y[4])
        self.time = 0.0
        self.terminated = self.truncated = False
        self.reason = ""
        return self.state

    def _derivative(self, y: Array, command: Array) -> Array:
        d, w = self.config.drone, self.config.world
        _, _, vx, vz, theta, omega, fl, fr = y
        total = fl + fr
        wind = d.wind_velocity
        derivative = np.array(
            [
                vx,
                vz,
                -total * np.sin(theta) / d.mass - d.linear_drag * (vx - wind[0]) / d.mass,
                total * np.cos(theta) / d.mass
                - w.gravity
                - d.linear_drag * (vz - wind[1]) / d.mass,
                omega,
                (d.arm * (fr - fl) - d.angular_drag * omega) / d.inertia,
                0.0,
                0.0,
            ]
        )
        if d.motor_tau:
            derivative[6:] = (command * np.asarray(d.thrust_gains) - y[6:]) / d.motor_tau
        return derivative

    def step(self, action) -> StepResult:
        if self.done:
            raise RuntimeError("Episode ended; call reset() before stepping again")
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (2,) or not np.isfinite(command).all():
            raise ValueError("Action must contain two finite force commands")
        d, w = self.config.drone, self.config.world
        command = np.clip(command, 0.0, d.max_motor_force)
        h = w.physics_dt
        for _ in range(round(w.control_dt / h)):
            if d.motor_tau == 0:
                self._y[6:] = command * np.asarray(d.thrust_gains)
            y = self._y
            k1 = self._derivative(y, command)
            k2 = self._derivative(y + h * k1 / 2, command)
            k3 = self._derivative(y + h * k2 / 2, command)
            k4 = self._derivative(y + h * k3, command)
            self._y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            self._y[4] = wrap_angle(self._y[4])
            self.time += h
            x, z = self._y[:2]
            r = w.collision_radius
            if not (r < x < w.width - r and r < z < w.height - r):
                self.terminated = True
                self.reason = "ground_contact" if z <= r else "boundary_contact"
                break
        self.truncated = not self.terminated and self.time >= w.max_seconds - 1e-9
        if self.truncated:
            self.reason = "time_limit"
        return StepResult(
            self.state,
            self.motor_forces,
            command.copy(),
            self.time,
            self.terminated,
            self.truncated,
            self.reason,
        )
