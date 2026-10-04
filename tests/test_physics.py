from dataclasses import replace

import numpy as np
import pytest

from skyjepa2d.config import DroneParams, SimConfig, WorldParams
from skyjepa2d.controllers import ManualController, WaypointController
from skyjepa2d.physics import PlanarDrone


def test_hover_is_equilibrium():
    plant = PlanarDrone()
    initial = plant.state
    for _ in range(200):
        plant.step(plant.hover_action)
    np.testing.assert_allclose(plant.state, initial, atol=1e-10)


def test_freefall_matches_analytic_solution():
    config = SimConfig(DroneParams(motor_tau=0, linear_drag=0, angular_drag=0))
    plant = PlanarDrone(config)
    plant.reset([4, 8, 0, 0, 0, 0], [0, 0])
    for _ in range(5):
        plant.step([0, 0])
    t = plant.time
    np.testing.assert_allclose(plant.state[1], 8 - 0.5 * config.world.gravity * t**2, atol=1e-9)
    np.testing.assert_allclose(plant.state[3], -config.world.gravity * t, atol=1e-9)


def test_tilt_redirects_thrust():
    config = SimConfig(DroneParams(motor_tau=0, linear_drag=0, angular_drag=0))
    plant = PlanarDrone(config)
    plant.reset([4, 8, 0, 0, 0.3, 0])
    plant.step(plant.hover_action)
    expected = -config.world.gravity * np.sin(0.3) * plant.time
    assert plant.state[2] == pytest.approx(expected, abs=1e-9)
    assert plant.state[3] < 0


def test_motor_lag_matches_first_order_response():
    plant = PlanarDrone()
    plant.reset(motor_forces=[0, 0])
    for _ in range(5):
        plant.step([6, 6])
    expected = 6 * (1 - np.exp(-plant.time / plant.config.drone.motor_tau))
    np.testing.assert_allclose(plant.motor_forces, expected, atol=1e-5)


def test_manual_right_tilts_and_moves_right():
    plant = PlanarDrone()
    manual = ManualController(plant.config)
    for _ in range(20):
        plant.step(manual.action(plant.state, steer=1, lift=0.1))
    assert plant.state[4] < 0
    assert plant.state[0] > 4


@pytest.mark.parametrize("goal", [[14, 6], [2, 2], [18, 10], [4, 9]])
def test_baseline_reaches_waypoints(goal):
    plant = PlanarDrone()
    controller = WaypointController(plant.config)
    for _ in range(240):
        plant.step(controller.action(plant.state, goal))
        assert not plant.terminated
    assert np.linalg.norm(plant.state[:2] - goal) < 0.08


def test_state_is_a_copy_and_nonfinite_actions_rejected():
    plant = PlanarDrone()
    state = plant.state
    state[0] = 999
    assert plant.state[0] == 4
    with pytest.raises(ValueError):
        plant.step([np.nan, 3])
    result = plant.step([-1, 999])
    np.testing.assert_array_equal(result.action, [0, 9])


def test_contacts_end_episode():
    plant = PlanarDrone(SimConfig(DroneParams(motor_tau=0)))
    plant.reset([4, 0.4, 0, -3, 0, 0], [0, 0])
    plant.step([0, 0])
    assert plant.terminated and plant.reason == "ground_contact"
    with pytest.raises(RuntimeError):
        plant.step([0, 0])


def test_fixed_steps_converge():
    states = []
    for h in (0.01, 0.005, 0.0025):
        config = SimConfig(DroneParams(motor_tau=0.05), WorldParams(physics_dt=h))
        plant = PlanarDrone(config)
        for k in range(20):
            plant.step([5 + 0.05 * np.sin(k), 5 + 0.03 * np.cos(k)])
        states.append(plant.state)
    assert np.linalg.norm(states[1] - states[2]) < np.linalg.norm(states[0] - states[2]) / 5


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        DroneParams(mass=0)
    with pytest.raises(ValueError):
        WorldParams(physics_dt=0.007)
    with pytest.raises(ValueError):
        SimConfig(drone=replace(DroneParams(), motor_tau=0.001))


def test_loaded_config_is_immutable_and_has_canonical_serialization():
    config = SimConfig.from_dict(SimConfig().to_dict())
    assert isinstance(config.drone.thrust_gains, tuple)
    assert config.to_dict() == SimConfig().to_dict()
    with pytest.raises(TypeError):
        config.drone.thrust_gains[0] = 0
