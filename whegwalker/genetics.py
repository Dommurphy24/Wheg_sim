from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import Literal

RobotType = Literal["wheg", "biped"]
ROBOT_TYPES: tuple[RobotType, ...] = ("wheg", "biped")

DEFAULT_LIMB_LENGTH = 46.0
TOTAL_LIMB_LENGTH = DEFAULT_LIMB_LENGTH * 3
LIMB_LENGTH_MIN = 30.0
LIMB_LENGTH_MAX = 65.0
MAX_LIMB_LENGTH_CHANGE = 8.0
DEFAULT_KNEE_ANGLE = math.radians(-45.0)
DEFAULT_ANKLE_ANGLE = math.radians(60.0)
KNEE_ANGLE_MIN = -1.35
KNEE_ANGLE_MAX = 1.35
ANKLE_ANGLE_MIN = -0.2
ANKLE_ANGLE_MAX = 1.65


@dataclass
class Genome:
    """Hub motor and oscillator parameters for the hip, knees, and ankles."""

    frequency: float
    hub_speed: float
    hip_amplitude: float
    hip_offset: float
    knee_amplitudes: list[float]
    knee_phases: list[float]
    knee_offset: float
    ankle_amplitudes: list[float]
    ankle_phases: list[float]
    ankle_offset: float
    limb_lengths: list[float]
    robot_type: RobotType = "wheg"
    hip_amplitudes: list[float] = field(default_factory=list)
    hip_phases: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.robot_type not in ROBOT_TYPES:
            raise ValueError(f"unsupported robot type: {self.robot_type}")
        leg_count = 3 if self.robot_type == "wheg" else 2
        if not self.hip_amplitudes:
            self.hip_amplitudes = [self.hip_amplitude] * leg_count
        if not self.hip_phases:
            self.hip_phases = (
                [0.0, math.pi]
                if self.robot_type == "biped"
                else [0.0] * leg_count
            )
        if len(self.hip_amplitudes) != leg_count or len(self.hip_phases) != leg_count:
            raise ValueError(
                f"{self.robot_type} genomes require {leg_count} hip amplitude and phase values"
            )

    @classmethod
    def random(
        cls, rng: random.Random, robot_type: RobotType = "wheg"
    ) -> Genome:
        if robot_type not in ROBOT_TYPES:
            raise ValueError(f"unsupported robot type: {robot_type}")
        leg_count = 3 if robot_type == "wheg" else 2
        hip_amplitude = rng.uniform(0.2, 0.9)
        return cls(
            frequency=rng.uniform(0.55, 2.2),
            hub_speed=(
                rng.choice((-1.0, 1.0)) * rng.uniform(1.0, 5.0)
                if robot_type == "wheg"
                else 0.0
            ),
            hip_amplitude=hip_amplitude,
            hip_offset=rng.uniform(-0.3, 0.3),
            knee_amplitudes=[rng.uniform(0.2, 0.9) for _ in range(leg_count)],
            knee_phases=[
                rng.uniform(-3.14159, 3.14159) for _ in range(leg_count)
            ],
            knee_offset=min(
                KNEE_ANGLE_MAX,
                max(
                    KNEE_ANGLE_MIN,
                    DEFAULT_KNEE_ANGLE + rng.uniform(-0.6, 0.6),
                ),
            ),
            ankle_amplitudes=[rng.uniform(0.2, 0.9) for _ in range(leg_count)],
            ankle_phases=[
                rng.uniform(-3.14159, 3.14159) for _ in range(leg_count)
            ],
            ankle_offset=min(
                ANKLE_ANGLE_MAX,
                max(
                    ANKLE_ANGLE_MIN,
                    DEFAULT_ANKLE_ANGLE + rng.uniform(-0.75, 0.75),
                ),
            ),
            limb_lengths=[DEFAULT_LIMB_LENGTH] * 3,
            robot_type=robot_type,
            hip_amplitudes=(
                [hip_amplitude] * leg_count
                if robot_type == "wheg"
                else [rng.uniform(0.2, 0.9) for _ in range(leg_count)]
            ),
            hip_phases=(
                [0.0] * leg_count
                if robot_type == "wheg"
                else [0.0, math.pi]
            ),
        )

    def copy(self) -> Genome:
        return Genome(
            frequency=self.frequency,
            hub_speed=self.hub_speed,
            hip_amplitude=self.hip_amplitude,
            hip_offset=self.hip_offset,
            knee_amplitudes=self.knee_amplitudes.copy(),
            knee_phases=self.knee_phases.copy(),
            knee_offset=self.knee_offset,
            ankle_amplitudes=self.ankle_amplitudes.copy(),
            ankle_phases=self.ankle_phases.copy(),
            ankle_offset=self.ankle_offset,
            limb_lengths=self.limb_lengths.copy(),
            robot_type=self.robot_type,
            hip_amplitudes=self.hip_amplitudes.copy(),
            hip_phases=self.hip_phases.copy(),
        )

    @classmethod
    def crossover(
        cls, first: Genome, second: Genome, rng: random.Random
    ) -> Genome:
        if first.robot_type != second.robot_type:
            raise ValueError("cannot cross genomes for different robot types")

        def choose(a: float, b: float) -> float:
            return a if rng.random() < 0.5 else b

        return cls(
            frequency=choose(first.frequency, second.frequency),
            hub_speed=choose(first.hub_speed, second.hub_speed),
            hip_amplitude=choose(first.hip_amplitude, second.hip_amplitude),
            hip_offset=choose(first.hip_offset, second.hip_offset),
            knee_amplitudes=[
                choose(first.knee_amplitudes[i], second.knee_amplitudes[i])
                for i in range(len(first.knee_amplitudes))
            ],
            knee_phases=[
                choose(first.knee_phases[i], second.knee_phases[i])
                for i in range(len(first.knee_phases))
            ],
            knee_offset=choose(first.knee_offset, second.knee_offset),
            ankle_amplitudes=[
                choose(first.ankle_amplitudes[i], second.ankle_amplitudes[i])
                for i in range(len(first.ankle_amplitudes))
            ],
            ankle_phases=[
                choose(first.ankle_phases[i], second.ankle_phases[i])
                for i in range(len(first.ankle_phases))
            ],
            ankle_offset=choose(first.ankle_offset, second.ankle_offset),
            limb_lengths=(
                first.limb_lengths.copy()
                if rng.random() < 0.5
                else second.limb_lengths.copy()
            ),
            robot_type=first.robot_type,
            hip_amplitudes=[
                choose(first.hip_amplitudes[i], second.hip_amplitudes[i])
                for i in range(len(first.hip_amplitudes))
            ],
            hip_phases=[
                choose(first.hip_phases[i], second.hip_phases[i])
                for i in range(len(first.hip_phases))
            ],
        )

    def mutate(self, rng: random.Random, rate: float = 0.18) -> None:
        def perturb(value: float, scale: float, low: float, high: float) -> float:
            if rng.random() < rate:
                value += rng.gauss(0.0, scale)
            return min(high, max(low, value))

        def perturb_angle(
            value: float, default: float, scale: float, low: float, high: float
        ) -> float:
            if rng.random() >= rate:
                return value
            mutation = rng.gauss(0.0, scale)
            neutral_bias = 0.05 * (default - value)
            return min(high, max(low, value + mutation + neutral_bias))

        self.frequency = perturb(self.frequency, 0.25, 0.4, 2.8)
        if self.robot_type == "wheg":
            self.hub_speed = perturb(self.hub_speed, 0.8, -6.0, 6.0)
            self.hip_amplitude = perturb(self.hip_amplitude, 0.18, 0.05, 1.2)
        else:
            self.hub_speed = 0.0
        self.hip_offset = perturb(self.hip_offset, 0.15, -0.65, 0.65)
        if self.robot_type == "biped":
            self.hip_amplitudes = [
                perturb(value, 0.25, 0.05, 1.2) for value in self.hip_amplitudes
            ]
            self.hip_phases = [
                perturb(value, 0.5, -3.14159, 3.14159)
                for value in self.hip_phases
            ]
        self.knee_amplitudes = [
            perturb(value, 0.25, 0.05, 1.2) for value in self.knee_amplitudes
        ]
        self.knee_phases = [
            perturb(value, 0.5, -3.14159, 3.14159)
            for value in self.knee_phases
        ]
        self.knee_offset = perturb_angle(
            self.knee_offset,
            DEFAULT_KNEE_ANGLE,
            0.3,
            KNEE_ANGLE_MIN,
            KNEE_ANGLE_MAX,
        )
        self.ankle_amplitudes = [
            perturb(value, 0.25, 0.05, 1.2) for value in self.ankle_amplitudes
        ]
        self.ankle_phases = [
            perturb(value, 0.5, -3.14159, 3.14159)
            for value in self.ankle_phases
        ]
        self.ankle_offset = perturb_angle(
            self.ankle_offset,
            DEFAULT_ANKLE_ANGLE,
            0.3,
            ANKLE_ANGLE_MIN,
            ANKLE_ANGLE_MAX,
        )
        if rng.random() < rate:
            donor, receiver = rng.sample(range(3), 2)
            transfer = max(
                -MAX_LIMB_LENGTH_CHANGE,
                min(MAX_LIMB_LENGTH_CHANGE, rng.gauss(0.0, 4.0)),
            )
            minimum_transfer = max(
                LIMB_LENGTH_MIN - self.limb_lengths[donor],
                self.limb_lengths[receiver] - LIMB_LENGTH_MAX,
            )
            maximum_transfer = min(
                LIMB_LENGTH_MAX - self.limb_lengths[donor],
                self.limb_lengths[receiver] - LIMB_LENGTH_MIN,
            )
            transfer = min(maximum_transfer, max(minimum_transfer, transfer))
            self.limb_lengths[donor] += transfer
            self.limb_lengths[receiver] -= transfer


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int = 16,
        elite_count: int | None = None,
        seed: int | None = None,
        robot_type: RobotType = "wheg",
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size must be at least 2")
        if elite_count is None:
            elite_count = min(2, population_size - 1)
        if not 1 <= elite_count < population_size:
            raise ValueError("elite_count must be between 1 and population_size - 1")
        self.rng = random.Random(seed)
        self.population_size = population_size
        self.elite_count = elite_count
        if robot_type not in ROBOT_TYPES:
            raise ValueError(f"unsupported robot type: {robot_type}")
        self.robot_type = robot_type
        self.population = [
            Genome.random(self.rng, robot_type) for _ in range(population_size)
        ]
        self.generation = 1
        self.best_fitness = float("-inf")
        self.best_genome: Genome | None = None

    def next_generation(self, scores: list[float]) -> None:
        if len(scores) != len(self.population):
            raise ValueError("one fitness score is required for every genome")

        ranked = sorted(
            zip(scores, self.population), key=lambda item: item[0], reverse=True
        )
        if ranked[0][0] > self.best_fitness:
            self.best_fitness = ranked[0][0]
            self.best_genome = ranked[0][1].copy()

        next_population = [genome.copy() for _, genome in ranked[: self.elite_count]]
        parent_pool = ranked[: max(self.elite_count, self.population_size // 2)]
        while len(next_population) < self.population_size:
            first = self.rng.choice(parent_pool)[1]
            second = self.rng.choice(parent_pool)[1]
            child = Genome.crossover(first, second, self.rng)
            child.mutate(self.rng)
            next_population.append(child)

        self.population = next_population
        self.generation += 1
