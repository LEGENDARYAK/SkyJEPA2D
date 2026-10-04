import numpy as np
import pytest

from skyjepa2d.config import SimConfig
from skyjepa2d.session import FlightSession, Mode


def test_autonomous_arrows_and_motor_cut_are_ignored():
    left, right = FlightSession(SimConfig()), FlightSession(SimConfig())
    left.toggle_pause()
    right.toggle_pause()
    for _ in range(30):
        left.tick(steer=1, lift=-1, cut_motors=True)
        right.tick()
    np.testing.assert_array_equal(left.plant.state, right.plant.state)


def test_waypoint_locks_even_when_paused():
    session = FlightSession(SimConfig())
    assert session.set_goal([10, 8])
    session.toggle_pause()
    assert not session.set_goal([2, 2])
    session.toggle_pause()
    assert not session.set_goal([2, 2])
    np.testing.assert_array_equal(session.goal, [10, 8])
    session.reset()
    assert session.set_goal([2, 2])


def test_manual_mode_consumes_arrows_and_mode_change_starts_new_episode():
    session = FlightSession(SimConfig(), Mode.MANUAL)
    session.toggle_pause()
    for _ in range(10):
        session.tick(steer=1, lift=0.2)
    assert session.plant.state[4] < 0
    session.set_mode(Mode.BASELINE)
    assert session.status == "ready" and not session.started
    np.testing.assert_array_equal(session.plant.state, [4, 4, 0, 0, 0, 0])


def test_learned_mode_requires_checkpoint():
    session = FlightSession(SimConfig())
    with pytest.raises(ValueError):
        session.set_mode(Mode.LEARNED)
    assert session.mode is Mode.BASELINE


def test_paused_simulation_does_not_advance():
    session = FlightSession(SimConfig())
    session.tick(steer=1, lift=1)
    assert session.plant.time == 0
