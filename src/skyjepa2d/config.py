"""Validated physical parameters. All quantities use SI units."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DroneParams:
    mass: float = 1.0
    inertia: float = 0.025
    arm: float = 0.22
    max_motor_force: float = 9.0
    motor_tau: float = 0.045
    linear_drag: float = 0.12
    angular_drag: float = 0.015
    thrust_gains: tuple[float, float] = (1.0, 1.0)
    wind_velocity: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        # JSON supplies lists; preserve immutability after loading a configuration.
        object.__setattr__(self, "thrust_gains", tuple(self.thrust_gains))
        object.__setattr__(self, "wind_velocity", tuple(self.wind_velocity))
        for name in ("mass", "inertia", "arm", "max_motor_force"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("motor_tau", "linear_drag", "angular_drag"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if len(self.thrust_gains) != 2 or any(
            not math.isfinite(v) or v <= 0 for v in self.thrust_gains
        ):
            raise ValueError("thrust_gains must contain two positive finite values")
        if len(self.wind_velocity) != 2 or not all(map(math.isfinite, self.wind_velocity)):
            raise ValueError("wind_velocity must contain two finite values")


@dataclass(frozen=True)
class WorldParams:
    gravity: float = 9.81
    physics_dt: float = 0.005
    control_dt: float = 0.05
    width: float = 20.0
    height: float = 12.0
    collision_radius: float = 0.25
    max_seconds: float = 30.0

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("World parameters must be finite and positive")
        ratio = self.control_dt / self.physics_dt
        if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-8):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        if 2 * self.collision_radius >= min(self.width, self.height):
            raise ValueError("collision_radius is too large for the flight area")


@dataclass(frozen=True)
class SimConfig:
    drone: DroneParams = field(default_factory=DroneParams)
    world: WorldParams = field(default_factory=WorldParams)

    def __post_init__(self) -> None:
        if self.drone.motor_tau and self.world.physics_dt > self.drone.motor_tau / 2:
            raise ValueError("physics_dt must be <= motor_tau / 2 for motor integration")

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, values: dict) -> SimConfig:
        if set(values) - {"drone", "world"}:
            raise ValueError("Config accepts only 'drone' and 'world' sections")
        return cls(DroneParams(**values.get("drone", {})), WorldParams(**values.get("world", {})))


def load_config(path: str | Path | None = None) -> SimConfig:
    if path is None:
        return SimConfig()
    return SimConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
