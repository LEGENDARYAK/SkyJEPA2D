"""Flight lifecycle and input isolation, independent of pygame."""

from __future__ import annotations

from enum import Enum

import numpy as np

from .config import SimConfig
from .controllers import ManualController, WaypointController
from .physics import PlanarDrone


class Mode(str, Enum):
    MANUAL = "manual"
    BASELINE = "baseline"
    LEARNED = "learned"


class FlightSession:
    """A navigation goal is editable only before the episode starts.

    Arrow input is consumed only by manual mode. Mode changes and resets start
    a new episode and reset controller history; human actions cannot enter a
    learned flight or the headless training collector.
    """

    def __init__(self, config: SimConfig, mode: Mode = Mode.BASELINE, learned=None):
        self.config = config
        self.plant = PlanarDrone(config)
        self.baseline = WaypointController(config)
        self.manual = ManualController(config)
        self.learned = learned
        self.mode = mode
        if mode is Mode.LEARNED and learned is None:
            raise ValueError("Learned mode requires a trained checkpoint")
        self.goal = np.array([0.7 * config.world.width, config.world.height / 2])
        self.reset()

    def reset(self) -> None:
        self.plant.reset()
        self.status = "ready"
        self.started = False
        self.arrival_time = 0.0
        self.last_action = self.plant.hover_action
        self.trail = [self.plant.state[:2]]
        if self.learned is not None:
            self.learned.reset(self.plant.state)

    def set_mode(self, mode: Mode) -> None:
        if mode is Mode.LEARNED and self.learned is None:
            raise ValueError("Load a checkpoint to use learned mode")
        self.mode = mode
        self.reset()

    def set_goal(self, goal) -> bool:
        if self.mode is Mode.MANUAL or self.started:
            return False
        goal = np.asarray(goal, dtype=float)
        w = self.config.world
        r = w.collision_radius + 0.2
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("Waypoint must contain two finite coordinates")
        if not (r < goal[0] < w.width - r and r < goal[1] < w.height - r):
            return False
        self.goal = goal.copy()
        return True

    def toggle_pause(self) -> None:
        if self.status == "running":
            self.status = "paused"
        elif self.status in ("ready", "paused"):
            self.started = True
            self.status = "running"

    def tick(self, steer=0.0, lift=0.0, cut_motors=False):
        if self.status != "running":
            return None
        state = self.plant.state
        if self.mode is Mode.MANUAL:
            action = self.manual.action(state, steer, lift, cut_motors)
        elif self.mode is Mode.BASELINE:
            action = self.baseline.action(state, self.goal)
        else:
            action = self.learned.action(state, self.goal)
        result = self.plant.step(action)
        self.last_action = result.action
        self.trail.append(result.state[:2])
        self.trail = self.trail[-1500:]
        if self.mode is Mode.LEARNED:
            self.learned.observe(result.state, result.action)
        if result.terminated:
            self.status = "crashed"
        elif result.truncated:
            self.status = "time_limit"
        elif self.mode is not Mode.MANUAL:
            arrived = (
                np.linalg.norm(result.state[:2] - self.goal) < 0.25
                and np.linalg.norm(result.state[2:4]) < 0.35
                and abs(result.state[4]) < 0.15
            )
            self.arrival_time = self.arrival_time + self.config.world.control_dt if arrived else 0
            if self.arrival_time >= 1.0:
                self.status = "arrived"
        return result
