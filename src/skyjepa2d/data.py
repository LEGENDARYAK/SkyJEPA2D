"""Headless trajectory collection and a versioned, time-aligned NPZ format.

An episode has N commands and N+1 states: actions[t] acts on states[t]
to produce states[t+1]. No keyboard, mouse, or display is imported here.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import SimConfig
from .controllers import WaypointController
from .physics import ACTION_NAMES, STATE_NAMES, PlanarDrone

SCHEMA_VERSION = 1


def randomized_config(config: SimConfig, rng: np.random.Generator) -> SimConfig:
    """Modest starter ranges; not the large transfer ranges in the paper."""
    d = config.drone
    params = replace(
        d,
        mass=d.mass * rng.uniform(0.85, 1.15),
        inertia=d.inertia * rng.uniform(0.8, 1.2),
        linear_drag=rng.uniform(0.05, 0.25),
        angular_drag=rng.uniform(0.008, 0.025),
        motor_tau=rng.uniform(0.025, 0.08),
        thrust_gains=tuple(rng.uniform(0.95, 1.05, size=2)),
        wind_velocity=tuple(rng.uniform(-0.6, 0.6, size=2)),
    )
    return SimConfig(params, config.world)


def collect_dataset(
    output: str | Path,
    config: SimConfig,
    episodes: int = 100,
    seconds: float = 12,
    seed: int = 42,
    randomize: bool = True,
    action_noise: float = 0.10,
) -> dict:
    if episodes < 10:
        raise ValueError("Use at least 10 episodes for independent train/validation/test splits")
    if not np.isfinite(seconds) or seconds < 2 or seconds > config.world.max_seconds:
        raise ValueError("seconds must be between 2 and the configured time limit")
    if not np.isfinite(action_noise) or action_noise < 0:
        raise ValueError("action_noise must be finite and nonnegative")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Dataset output must be empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(episodes)
    n_val = max(1, round(0.1 * episodes))
    val_ids = set(permutation[:n_val].tolist())
    test_ids = set(permutation[n_val : 2 * n_val].tolist())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "config": config.to_dict(),
        "control_dt": config.world.control_dt,
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
        "action_units": "N",
        "randomized": randomize,
        "collector": "baseline_pd_with_correlated_force_excitation",
        "episodes": [],
    }
    steps = int(seconds / config.world.control_dt)
    for episode_id in range(episodes):
        episode_seed = int(rng.integers(0, 2**31 - 1))
        erng = np.random.default_rng(episode_seed)
        domain = randomized_config(config, erng) if randomize else config
        plant = PlanarDrone(domain)
        w = domain.world
        initial = [
            erng.uniform(0.2, 0.8) * w.width,
            erng.uniform(0.25, 0.65) * w.height,
            *erng.uniform(-0.4, 0.4, size=2),
            erng.uniform(-0.12, 0.12),
            erng.uniform(-0.2, 0.2),
        ]
        plant.reset(initial)
        # The collecting controller may know the domain; the learned model never receives it.
        controller = WaypointController(domain, max_speed=erng.uniform(1.4, 3.0))
        goal = plant.state[:2].copy()
        states, motors, times = [plant.state], [plant.motor_forces], [plant.time]
        actions, goals = [], []
        noise = np.zeros(2)
        waypoint_interval = max(1, round(3.0 / w.control_dt))
        for step in range(steps):
            if step % waypoint_interval == 0:
                goal = np.array(
                    [
                        erng.uniform(0.2, 0.8) * w.width,
                        erng.uniform(0.25, 0.75) * w.height,
                    ]
                )
            noise = 0.8 * noise + erng.normal(0, action_noise, size=2)
            command = controller.action(plant.state, goal) + noise
            result = plant.step(command)
            actions.append(result.action)
            goals.append(goal.copy())
            states.append(result.state)
            motors.append(result.motor_forces)
            times.append(result.time)
            if plant.done:
                break
        split = (
            "validation" if episode_id in val_ids else "test" if episode_id in test_ids else "train"
        )
        filename = f"episode_{episode_id:05d}.npz"
        np.savez_compressed(
            output / filename,
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            motor_forces=np.asarray(motors, dtype=np.float32),
            times=np.asarray(times, dtype=np.float64),
            goals=np.asarray(goals, dtype=np.float32),
        )
        manifest["episodes"].append(
            {
                "file": filename,
                "split": split,
                "seed": episode_seed,
                "steps": len(actions),
                "terminal": plant.terminated,
                "reason": plant.reason or "collection_limit",
                "config": domain.to_dict(),
            }
        )
        if (episode_id + 1) % max(1, episodes // 10) == 0:
            print(f"Collected {episode_id + 1}/{episodes} episodes", flush=True)
    manifest["status"] = "complete"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_manifest(directory: str | Path) -> dict:
    manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise ValueError("Unsupported or incomplete dataset")
    return manifest
