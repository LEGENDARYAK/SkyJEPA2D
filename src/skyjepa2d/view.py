"""Pygame visualization; renderer also supports off-screen preview export."""

from __future__ import annotations

import math
import os
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import pygame

from .session import FlightSession, Mode

BG = (12, 19, 31)
PANEL = (20, 30, 46)
GRID = (34, 48, 65)
TEXT = (225, 235, 244)
MUTED = (144, 166, 187)
ACCENTS = {
    Mode.MANUAL: (80, 217, 187),
    Mode.BASELINE: (128, 178, 255),
    Mode.LEARNED: (214, 171, 255),
}
GOLD = (255, 202, 112)


class Renderer:
    """Render an existing session; drawing cannot mutate the simulation."""

    def __init__(self, config, size=(1100, 740)):
        pygame.font.init()
        self.config = config
        self.size = size
        self.fonts = {n: pygame.font.SysFont("Arial", n) for n in (13, 15, 17, 20, 27)}
        self.flight = pygame.Rect(28, 146, 744, 446)
        self.buttons = {mode: pygame.Rect(28 + i * 190, 89, 178, 38) for i, mode in enumerate(Mode)}

    def text(self, surface, content, point, size=17, color=TEXT):
        surface.blit(self.fonts[size].render(str(content), True, color), point)

    def to_screen(self, position):
        w = self.config.world
        return (
            round(self.flight.left + position[0] / w.width * self.flight.width),
            round(self.flight.bottom - position[1] / w.height * self.flight.height),
        )

    def to_world(self, position):
        w = self.config.world
        return np.array(
            [
                (position[0] - self.flight.left) / self.flight.width * w.width,
                (self.flight.bottom - position[1]) / self.flight.height * w.height,
            ]
        )

    def draw(self, surface, session, toast="", controller_ms=None):
        surface.fill(BG)
        accent = ACCENTS[session.mode]
        self.text(surface, "SKYJEPA", (28, 23), 27, accent)
        self.text(surface, "/  PLANAR FLIGHT LAB", (184, 31), 17, TEXT)
        self.text(
            surface,
            "A small physics world for learning how actions become motion.",
            (28, 59),
            15,
            MUTED,
        )
        for index, (mode, rect) in enumerate(self.buttons.items(), 1):
            selected = mode == session.mode
            pygame.draw.rect(surface, PANEL, rect, border_radius=7)
            pygame.draw.rect(
                surface,
                ACCENTS[mode] if selected else GRID,
                rect,
                2 if selected else 1,
                border_radius=7,
            )
            self.text(
                surface,
                f"{index}   {mode.value.upper()}",
                (rect.x + 15, rect.y + 10),
                15,
                ACCENTS[mode] if selected else MUTED,
            )
        self.text(surface, "STATE INPUTS  /  TWO MOTOR GROUPS", (631, 104), 13, MUTED)
        pygame.draw.rect(surface, PANEL, self.flight, border_radius=8)
        w = self.config.world
        surface.set_clip(self.flight)
        for x in np.arange(0, w.width + 1e-8, 2):
            px, _ = self.to_screen([x, 0])
            pygame.draw.line(surface, GRID, (px, self.flight.top), (px, self.flight.bottom))
        for z in np.arange(0, w.height + 1e-8, 2):
            _, py = self.to_screen([0, z])
            pygame.draw.line(surface, GRID, (self.flight.left, py), (self.flight.right, py))
        if len(session.trail) > 1:
            points = [self.to_screen(p) for p in session.trail]
            pygame.draw.lines(surface, tuple(int(c * 0.6) for c in accent), False, points, 2)
        if session.mode is not Mode.MANUAL:
            center = self.to_screen(session.goal)
            pygame.draw.circle(surface, GOLD, center, 13, 2)
            pygame.draw.circle(surface, GOLD, center, 4)
            for dx, dy in [(19, 0), (-19, 0), (0, 19), (0, -19)]:
                pygame.draw.line(
                    surface,
                    GOLD,
                    (center[0] + dx, center[1] + dy),
                    (center[0] + dx * 1.4, center[1] + dy * 1.4),
                    2,
                )
            label_x = center[0] + 42 if center[0] < self.flight.right - 150 else center[0] - 100
            label_y = max(self.flight.top + 10, center[1] - 30)
            self.text(surface, "WAYPOINT", (label_x, label_y), 13, GOLD)
        state = session.plant.state
        center = np.asarray(self.to_screen(state[:2]), dtype=float)
        theta = state[4]
        c, s = math.cos(theta), math.sin(theta)

        def local(x, z):
            return (center + np.array([c * x - s * z, -s * x - c * z])).astype(int)

        pygame.draw.line(surface, accent, local(-26, 0), local(26, 0), 4)
        pygame.draw.polygon(
            surface, accent, [local(-13, -7), local(13, -7), local(13, 7), local(-13, 7)]
        )
        pygame.draw.line(surface, TEXT, local(4, 0), local(13, 0), 2)
        for i, rotor_x in enumerate((-26, 26)):
            pygame.draw.line(surface, TEXT, local(rotor_x - 10, 3), local(rotor_x + 10, 3), 3)
            length = 6 + 18 * session.plant.motor_forces[i] / self.config.drone.max_motor_force
            pygame.draw.line(surface, GOLD, local(rotor_x, -5), local(rotor_x, -length), 2)
        self.text(
            surface,
            "x / horizontal position",
            (self.flight.left + 13, self.flight.bottom - 25),
            13,
            MUTED,
        )
        self.text(surface, "z / altitude", (self.flight.left + 13, self.flight.top + 12), 13, MUTED)
        surface.set_clip(None)
        pygame.draw.rect(surface, GRID, self.flight, 1, border_radius=8)
        sidebar = pygame.Rect(795, 146, 277, 446)
        pygame.draw.rect(surface, PANEL, sidebar, border_radius=8)
        self.text(surface, "FLIGHT TELEMETRY", (813, 166), 15, MUTED)
        self.text(surface, session.status.replace("_", " ").upper(), (813, 193), 20, accent)
        fields = [
            ("Simulation time", f"{session.plant.time:5.2f} s"),
            ("Position x / z", f"{state[0]:.2f} / {state[1]:.2f} m"),
            ("Velocity x / z", f"{state[2]:.2f} / {state[3]:.2f} m/s"),
            ("Pitch", f"{math.degrees(state[4]):+.1f} deg"),
            ("Pitch rate", f"{math.degrees(state[5]):+.1f} deg/s"),
        ]
        if session.mode is not Mode.MANUAL:
            fields.append(
                ("Waypoint distance", f"{np.linalg.norm(state[:2] - session.goal):.2f} m")
            )
        for i, (label, value) in enumerate(fields):
            y = 234 + i * 36
            self.text(surface, label, (813, y), 13, MUTED)
            self.text(surface, value, (813, y + 15), 15)
        y0 = 462
        self.text(surface, "APPLIED MOTOR FORCES", (813, y0), 13, MUTED)
        for i, label in enumerate(("L", "R")):
            y = y0 + 24 + i * 29
            force = session.plant.motor_forces[i]
            self.text(surface, label, (813, y), 15, accent)
            pygame.draw.rect(surface, GRID, (836, y + 3, 155, 10), border_radius=3)
            pygame.draw.rect(
                surface,
                accent,
                (
                    836,
                    y + 3,
                    round(155 * np.clip(force / self.config.drone.max_motor_force, 0, 1)),
                    10,
                ),
                border_radius=3,
            )
            self.text(surface, f"{force:.1f} N", (1000, y), 13)
        hz = 1 / self.config.world.control_dt
        timing = "--" if controller_ms is None else f"{controller_ms:.1f}"
        self.text(surface, f"Control {hz:g} Hz  |  compute {timing} ms", (813, 568), 13, MUTED)
        pygame.draw.rect(surface, PANEL, (28, 613, 1044, 101), border_radius=8)
        if session.mode is Mode.MANUAL:
            self.text(surface, "YOU COMMAND THRUST AND LEAN", (46, 629), 17, accent)
            self.text(
                surface,
                "Up / Down: lift     Left / Right: tilt     Space: cut motors",
                (46, 655),
                15,
            )
            self.text(
                surface,
                "Release Left / Right to level. Gravity and momentum still apply.",
                (46, 679),
                13,
                MUTED,
            )
        else:
            self.text(
                surface, "CHOOSE A WAYPOINT, THEN LET THE CONTROLLER FLY", (46, 629), 17, accent
            )
            line = "Click the flight area before starting. Enter launches the flight."
            if session.started:
                line = "Waypoint locked for this episode. Arrow keys and clicks are ignored."
            self.text(surface, line, (46, 655), 15)
            label = (
                "Conventional PD baseline" if session.mode is Mode.BASELINE else "JEPA model + MPPI"
            )
            self.text(surface, label + "  |  R starts a fresh episode.", (46, 679), 13, MUTED)
        self.text(
            surface,
            "ENTER start / pause     R reset     1 / 2 / 3 modes     ESC exit",
            (638, 30),
            13,
            MUTED,
        )
        if toast:
            pygame.draw.rect(surface, BG, (55, 539, 690, 37), border_radius=6)
            self.text(surface, toast, (68, 550), 15, GOLD)


def run_app(config, mode=Mode.BASELINE, learned=None) -> None:
    pygame.init()
    try:
        renderer = Renderer(config)
        screen = pygame.display.set_mode(renderer.size)
        pygame.display.set_caption("SkyJEPA 2D - Planar Flight Lab")
        session = FlightSession(config, mode, learned)
        clock = pygame.time.Clock()
        accumulator = 0.0
        toast, toast_until, controller_ms = "", 0.0, 0.0
        running = True
        while running:
            frame_seconds = min(clock.tick(60) / 1000, 0.15)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_RETURN, pygame.K_p):
                        session.toggle_pause()
                    elif event.key == pygame.K_r:
                        session.reset()
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        choice = {
                            pygame.K_1: Mode.MANUAL,
                            pygame.K_2: Mode.BASELINE,
                            pygame.K_3: Mode.LEARNED,
                        }[event.key]
                        try:
                            session.set_mode(choice)
                        except ValueError as exc:
                            toast, toast_until = str(exc), time.monotonic() + 3
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for button_mode, rect in renderer.buttons.items():
                        if rect.collidepoint(event.pos):
                            try:
                                session.set_mode(button_mode)
                            except ValueError as exc:
                                toast, toast_until = str(exc), time.monotonic() + 3
                    if renderer.flight.collidepoint(event.pos) and session.mode is not Mode.MANUAL:
                        if not session.set_goal(renderer.to_world(event.pos)):
                            toast = (
                                "Waypoint locked or outside flight margins. R resets the episode."
                            )
                            toast_until = time.monotonic() + 3
            if session.status == "running":
                accumulator = min(accumulator + frame_seconds, 3 * config.world.control_dt)
                keys = pygame.key.get_pressed()
                while accumulator >= config.world.control_dt:
                    before = time.perf_counter()
                    session.tick(
                        steer=int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT]),
                        lift=int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN]),
                        cut_motors=bool(keys[pygame.K_SPACE]),
                    )
                    controller_ms = (time.perf_counter() - before) * 1000
                    accumulator -= config.world.control_dt
            else:
                accumulator = 0.0
            renderer.draw(
                screen, session, toast if time.monotonic() < toast_until else "", controller_ms
            )
            pygame.display.flip()
    finally:
        pygame.quit()
