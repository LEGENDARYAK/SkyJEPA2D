import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from skyjepa2d.config import SimConfig
from skyjepa2d.session import FlightSession
from skyjepa2d.view import Renderer


def test_rendering_is_read_only_and_coordinates_round_trip():
    renderer = Renderer(SimConfig())
    surface = pygame.Surface(renderer.size)
    session = FlightSession(SimConfig())
    initial = session.plant.state
    renderer.draw(surface, session)
    np.testing.assert_array_equal(session.plant.state, initial)
    point = np.array([12.0, 7.0])
    np.testing.assert_allclose(renderer.to_world(renderer.to_screen(point)), point, atol=0.03)
    assert len(set(pygame.image.tobytes(surface, "RGB"))) > 100


def test_app_event_loop_mode_changes_and_goal_lock(monkeypatch):
    from collections import deque

    from skyjepa2d.session import Mode
    from skyjepa2d.view import run_app

    renderer = Renderer(SimConfig())
    frames = deque(
        [
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)],
            [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=renderer.to_screen([7, 8]))],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2)],
            [pygame.event.Event(pygame.QUIT)],
        ]
    )
    snapshots = []
    original_draw = Renderer.draw

    class FixedClock:
        def tick(self, fps):
            return 50

    class Pressed:
        def __getitem__(self, key):
            return key in (pygame.K_RIGHT, pygame.K_UP, pygame.K_SPACE)

    def record(self, surface, session, *args):
        snapshots.append(
            (
                session.mode,
                session.status,
                session.plant.time,
                session.goal.copy(),
                session.plant.state.copy(),
            )
        )
        original_draw(self, surface, session, *args)

    monkeypatch.setattr(pygame.event, "get", lambda: frames.popleft())
    monkeypatch.setattr(pygame.time, "Clock", FixedClock)
    monkeypatch.setattr(pygame.key, "get_pressed", Pressed)
    monkeypatch.setattr(Renderer, "draw", record)
    run_app(SimConfig())
    # A live click cannot change the goal, nor can pause unlock it.
    np.testing.assert_array_equal(snapshots[0][3], snapshots[1][3])
    assert snapshots[2][1] == "paused"
    assert snapshots[2][2] == snapshots[1][2]
    assert snapshots[3][0] is Mode.MANUAL and snapshots[3][2] == 0
    assert snapshots[4][0] is Mode.MANUAL and snapshots[4][2] > 0
    assert snapshots[5][0] is Mode.BASELINE and snapshots[5][2] == 0
