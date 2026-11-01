from __future__ import annotations

import math

import pygame
import pymunk

from whegwalker.genetics import Genome
from whegwalker.physics import WhegSimulation
from whegwalker.trainer import EvolutionWorker

WIDTH = 1100
HEIGHT = 720
PANEL_WIDTH = 300
BACKGROUND = (21, 25, 34)
PANEL = (30, 36, 48)
TEXT = (235, 239, 245)
MUTED = (155, 166, 182)
ACCENT = (86, 196, 180)
ORANGE = (255, 170, 90)
GROUND = (87, 100, 116)
VIEW_WIDTH = WIDTH - PANEL_WIDTH
TRACK_MARK_SPACING = 100


class SimulatorApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Wheg Walker - Genetic Algorithm Lab")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 18)
        self.small_font = pygame.font.SysFont("segoeui", 14)
        self.title_font = pygame.font.SysFont("segoeui", 24, bold=True)
        self.evolution = EvolutionWorker()
        self.evolution.start()
        self.training = True
        self.best_simulation: WhegSimulation | None = None
        self.displayed_genome: Genome | None = None
        self.speed = 1
        self.camera_x = 0.0
        self.running = True

    def run(self) -> None:
        try:
            while self.running:
                self._events()
                self._update()
                self._draw()
                pygame.display.flip()
                self.clock.tick(60)
        finally:
            self.evolution.stop()
            pygame.quit()

    def _events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.training = not self.training
                    self.evolution.set_paused(not self.training)
                elif event.key == pygame.K_r:
                    self.evolution.stop()
                    self.evolution = EvolutionWorker()
                    self.evolution.set_paused(not self.training)
                    self.evolution.start()
                    self.best_simulation = None
                    self.displayed_genome = None
                elif event.key == pygame.K_1:
                    self.speed = 1
                elif event.key == pygame.K_2:
                    self.speed = 2
                elif event.key == pygame.K_3:
                    self.speed = 4
                elif event.key == pygame.K_b and self.displayed_genome is not None:
                    self.best_simulation = WhegSimulation(self.displayed_genome)

    def _update(self) -> None:
        status = self.evolution.poll()
        best_genome = status["best_genome"]
        if best_genome is not None and best_genome != self.displayed_genome:
            self.displayed_genome = best_genome.copy()
            self.best_simulation = WhegSimulation(self.displayed_genome)

        if self.best_simulation is not None:
            self.best_simulation.step(self.speed)
            if self.best_simulation.fallen and self.displayed_genome is not None:
                self.best_simulation = WhegSimulation(self.displayed_genome)

        if self.best_simulation is not None:
            target_x = self.best_simulation.torso.position.x
            view_width = WIDTH - PANEL_WIDTH
            self.camera_x = max(0.0, target_x - view_width * 0.36)

    def _world_to_screen(self, point: pymunk.Vec2d) -> tuple[int, int]:
        return (int(point.x - self.camera_x), int(HEIGHT - point.y))

    def _draw(self) -> None:
        self.screen.fill(BACKGROUND)
        simulation = self.best_simulation
        floor_y = HEIGHT - 32
        self._draw_environment(floor_y, simulation)
        if simulation is not None:
            self._draw_robot(simulation, floor_y)
            self._draw_joint_status(simulation)
        else:
            message = self.small_font.render(
                "Starting evolution worker...", True, MUTED
            )
            self.screen.blit(message, (30, 30))
        self._draw_panel()

    def _draw_environment(
        self, floor_y: int, simulation: WhegSimulation | None
    ) -> None:
        pygame.draw.rect(self.screen, (36, 43, 55), (0, 0, VIEW_WIDTH, HEIGHT))
        pygame.draw.rect(
            self.screen, (29, 35, 45), (0, floor_y, VIEW_WIDTH, HEIGHT - floor_y)
        )
        pygame.draw.line(
            self.screen, GROUND, (0, floor_y), (VIEW_WIDTH, floor_y), 4
        )
        if simulation is not None:
            first_marker = math.floor(self.camera_x / TRACK_MARK_SPACING)
            marker_count = math.ceil(VIEW_WIDTH / TRACK_MARK_SPACING) + 2
            for index in range(first_marker, first_marker + marker_count):
                world_x = index * TRACK_MARK_SPACING
                screen_x = int(world_x - self.camera_x)
                distance = round(world_x - simulation.start_x)
                pygame.draw.line(
                    self.screen,
                    (71, 84, 101),
                    (screen_x, floor_y - 18),
                    (screen_x, floor_y + 1),
                    2,
                )
                marker_label = self.small_font.render(
                    f"{distance:+d}", True, MUTED
                )
                self.screen.blit(
                    marker_label, (screen_x + 4, floor_y - 40)
                )

    def _draw_robot(self, simulation: WhegSimulation, floor_y: int) -> None:
        lengths = simulation.limb_lengths
        for index, (hip, shin, foot) in enumerate(
            zip(simulation.hips, simulation.shins, simulation.feet)
        ):
            color = ACCENT if index == 0 else (110, 165, 205)
            hip_start = hip.local_to_world((-lengths[0] / 2, 0))
            hip_end = hip.local_to_world((lengths[0] / 2, 0))
            pygame.draw.line(
                self.screen,
                color,
                self._world_to_screen(hip_start),
                self._world_to_screen(hip_end),
                10,
            )
            pygame.draw.circle(
                self.screen, (222, 230, 238), self._world_to_screen(hip_start), 6
            )
            knee = shin.local_to_world((-lengths[1] / 2, 0))
            ankle = shin.local_to_world((lengths[1] / 2, 0))
            toe = foot.local_to_world((lengths[2] / 2, 0))
            pygame.draw.line(
                self.screen,
                ORANGE,
                self._world_to_screen(knee),
                self._world_to_screen(ankle),
                10,
            )
            pygame.draw.line(
                self.screen,
                (205, 169, 112),
                self._world_to_screen(ankle),
                self._world_to_screen(toe),
                10,
            )
            pygame.draw.circle(
                self.screen, (255, 220, 164), self._world_to_screen(toe), 5
            )

        center = self._world_to_screen(simulation.torso.position)
        pygame.draw.circle(self.screen, (212, 221, 234), center, 24)
        pygame.draw.circle(self.screen, (64, 78, 96), center, 18)
        pygame.draw.circle(self.screen, ACCENT, center, 5)
        heading = simulation.torso.local_to_world((26.0, 0.0))
        pygame.draw.line(
            self.screen, (255, 255, 255), center, self._world_to_screen(heading), 3
        )

        delta = simulation.torso.position.x - simulation.start_x
        bar_x = center[0]
        bar_y = floor_y - 26
        bar_width = int(max(-80, min(80, delta * 2.0)))
        pygame.draw.line(self.screen, (74, 84, 100), (bar_x, bar_y), (bar_x + 170, bar_y), 1)
        pygame.draw.line(
            self.screen,
            ACCENT,
            (bar_x, bar_y - 4),
            (bar_x + bar_width, bar_y - 4),
            4,
        )

    def _draw_joint_status(self, simulation: WhegSimulation) -> None:
        hip_angle = math.degrees(
            simulation.hips[0].angle
            - simulation.torso.angle
            - simulation.hip_initial_angle
        )
        knee_angles = [
            math.degrees(shin.angle - hip.angle)
            for hip, shin in zip(simulation.hips, simulation.shins)
        ]
        ankle_angles = [
            math.degrees(foot.angle - shin.angle)
            for shin, foot in zip(simulation.shins, simulation.feet)
        ]
        lines = (
            f"Hub: {math.degrees(simulation.torso.angle) % 360:.0f} deg"
            f"  {simulation.torso.angular_velocity:+.2f} rad/s"
            f"  target {simulation.genome.hub_speed:+.2f}",
            f"Shared hip: {hip_angle:+.0f} deg"
            f"  knees: {knee_angles[0]:+.0f}, {knee_angles[1]:+.0f}, {knee_angles[2]:+.0f} deg",
            f"ankles: {ankle_angles[0]:+.0f}, {ankle_angles[1]:+.0f}, {ankle_angles[2]:+.0f} deg",
        )
        for index, line in enumerate(lines):
            label = self.small_font.render(line, True, MUTED)
            self.screen.blit(label, (20, 18 + index * 21))

    def _draw_panel(self) -> None:
        x = WIDTH - PANEL_WIDTH
        pygame.draw.rect(self.screen, PANEL, (x, 0, PANEL_WIDTH, HEIGHT))
        cursor = x + 24

        def text(value: str, y: int, color=TEXT, font=None) -> None:
            self.screen.blit(
                (font or self.font).render(value, True, color), (x + 24, y)
            )

        text("WHEG WALKER", 24, ACCENT, pygame.font.SysFont("segoeui", 24, bold=True))
        text("Genetic algorithm lab", 58, MUTED, self.small_font)
        pygame.draw.line(self.screen, (62, 72, 88), (cursor, 88), (WIDTH - 24, 88), 1)

        status = self.evolution.status
        genome = self.displayed_genome
        text(f"Generation: {status['generation']}", 108)
        text(
            f"Evaluated: {status['evaluated']} / {status['population_size']}",
            136,
            MUTED,
        )
        best = status["best_fitness"]
        best_label = (
            f"{best:.1f}"
            if best is not None and math.isfinite(best)
            else "evaluating..."
        )
        text(f"Best fitness: {best_label}", 164, ACCENT)
        generation_best = status["generation_best"]
        if status["error"]:
            text(
                f"Worker error: {status['error'][:34]}",
                192,
                (255, 130, 130),
                self.small_font,
            )
        elif generation_best is not None:
            text(f"Generation best: {generation_best:.1f}", 192, MUTED)

        pygame.draw.line(self.screen, (62, 72, 88), (cursor, 224), (WIDTH - 24, 224), 1)
        text("CURRENT GAIT", 242, ORANGE, self.small_font)
        if genome is None:
            text("Waiting for first trial...", 269, MUTED, self.small_font)
        else:
            text(f"Frequency: {genome.frequency:.2f} Hz", 269)
            text(
                f"Hub motor: {genome.hub_speed:+.2f} rad/s",
                295,
                ACCENT,
                self.small_font,
            )
            text(
                f"Shared hip amplitude: {genome.hip_amplitude:.2f} rad",
                319,
                MUTED,
                self.small_font,
            )
            text(
                "Knee amplitudes: "
                + " / ".join(f"{amplitude:.2f}" for amplitude in genome.knee_amplitudes),
                345,
                MUTED,
                self.small_font,
            )
            text(
                f"Shared knee bias: {math.degrees(genome.knee_offset):+.0f} deg",
                367,
                MUTED,
                self.small_font,
            )
            text(
                "Ankle amplitudes: "
                + " / ".join(f"{amplitude:.2f}" for amplitude in genome.ankle_amplitudes),
                389,
                MUTED,
                self.small_font,
            )
            text(
                f"Shared ankle bias: {math.degrees(genome.ankle_offset):+.0f} deg",
                411,
                MUTED,
                self.small_font,
            )
            text(
                "Link lengths U/M/F: "
                + " / ".join(f"{length:.0f}" for length in genome.limb_lengths)
                + f" (total {sum(genome.limb_lengths):.0f})",
                429,
                MUTED,
                self.small_font,
            )

        chart_top = 450
        pygame.draw.line(
            self.screen, (62, 72, 88), (cursor, chart_top), (WIDTH - 24, chart_top), 1
        )
        text("BEST FITNESS BY GENERATION", chart_top + 14, MUTED, self.small_font)
        self._draw_chart((x + 24, chart_top + 42, PANEL_WIDTH - 48, 80))

        pygame.draw.line(
            self.screen, (62, 72, 88), (cursor, 568), (WIDTH - 24, 568), 1
        )
        text("CONTROLS", 582, ACCENT, self.small_font)
        text("Space  pause / resume training", 608, MUTED, self.small_font)
        text("B  replay best evolved gait", 630, MUTED, self.small_font)
        text("1 / 2 / 3  playback speed", 652, MUTED, self.small_font)
        text("R  restart evolution", 674, MUTED, self.small_font)
        error = status["error"]
        state = "ERROR" if error else "TRAINING" if self.training else "PAUSED"
        badge = self.small_font.render(state, True, TEXT)
        badge_rect = badge.get_rect(topright=(WIDTH - 22, 30))
        badge_color = (
            (140, 55, 55)
            if error
            else (38, 101, 88) if self.training else (79, 84, 95)
        )
        pygame.draw.rect(
            self.screen, badge_color, badge_rect.inflate(18, 10), border_radius=8
        )
        self.screen.blit(badge, badge_rect)

    def _draw_chart(self, rect: tuple[int, int, int, int]) -> None:
        x, y, width, height = rect
        pygame.draw.rect(self.screen, (23, 28, 38), rect, border_radius=4)
        history = self.evolution.status["history"][-40:]
        if len(history) < 2:
            return
        low = min(history)
        span = max(1.0, max(history) - low)
        points = [
            (
                x + int(index * width / (len(history) - 1)),
                y + height - int((value - low) / span * (height - 12)) - 6,
            )
            for index, value in enumerate(history)
        ]
        pygame.draw.lines(self.screen, ACCENT, False, points, 2)


def main() -> None:
    SimulatorApp().run()
