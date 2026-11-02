from __future__ import annotations

import math
from dataclasses import dataclass

import pymunk

from whegwalker.genetics import (
    ANKLE_ANGLE_MAX,
    ANKLE_ANGLE_MIN,
    DEFAULT_ANKLE_ANGLE,
    DEFAULT_KNEE_ANGLE,
    DEFAULT_LIMB_LENGTH,
    KNEE_ANGLE_MAX,
    KNEE_ANGLE_MIN,
    Genome,
)


@dataclass(frozen=True)
class FitnessWeights:
    max_speed: float = 40.0
    forward_distance: float = 20.0
    low_bounce: float = 20.0
    low_energy: float = 20.0

    def __post_init__(self) -> None:
        weights = (
            self.max_speed,
            self.forward_distance,
            self.low_bounce,
            self.low_energy,
        )
        if any(
            not math.isfinite(weight) or not 0.0 <= weight <= 100.0
            for weight in weights
        ):
            raise ValueError("Fitness weights must be finite values between 0 and 100")
        if not any(weights):
            raise ValueError("At least one fitness weight must be greater than zero")


DEFAULT_FITNESS_WEIGHTS = FitnessWeights()


class WhegSimulation:
    """One deterministic-time trial of the three-spoke articulated robot."""

    DT = 1.0 / 120.0
    PHYSICS_SUBSTEPS = 4
    SOLVER_ITERATIONS = 25
    GROUND_HALF_LENGTH = 1_000_000.0
    HIP_LENGTH = 46.0
    SHIN_LENGTH = 46.0
    FOOT_LENGTH = 46.0
    HIP_LIMIT = 1.25
    KNEE_LIMIT = max(abs(KNEE_ANGLE_MIN), abs(KNEE_ANGLE_MAX))
    ANKLE_MIN = ANKLE_ANGLE_MIN
    ANKLE_MAX = ANKLE_ANGLE_MAX
    GRAVITY = 900.0
    TORSO_MASS = 4.0
    HIP_MASS = 0.8
    SHIN_MASS = 0.55
    FOOT_MASS = 0.4
    MOTOR_TORQUE_LIMIT = 750_000.0
    HUB_TORQUE_LIMIT = 2_000_000.0
    JOINT_SPRING_STIFFNESS = 300.0
    JOINT_SPRING_DAMPING = 300.0
    JOINT_SPRING_TORQUE_LIMIT = 50_000.0
    SPEED_REFERENCE = 200.0
    BOUNCE_REFERENCE = 80.0
    POWER_REFERENCE = 1_500_000.0
    FALL_PENALTY = 1.0

    def __init__(self, genome: Genome, start_x: float = 120.0) -> None:
        self.genome = genome.copy()
        self.limb_lengths = self.genome.limb_lengths.copy()
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -self.GRAVITY)
        self.space.damping = 0.995
        self.space.iterations = self.SOLVER_ITERATIONS
        self.ground_y = 32.0
        self.elapsed = 0.0
        self.start_x = start_x
        self.fallen = False
        self.max_forward_speed = 0.0
        self.min_torso_y = 160.0
        self.max_torso_y = 160.0
        self.motor_energy = 0.0

        self.ground = pymunk.Segment(
            self.space.static_body,
            (-self.GROUND_HALF_LENGTH, self.ground_y),
            (self.GROUND_HALF_LENGTH, self.ground_y),
            4.0,
        )
        self.ground.friction = 1.2
        self.ground.elasticity = 0.0
        self.space.add(self.ground)

        center = pymunk.Vec2d(start_x, 160.0)
        self.torso = pymunk.Body(
            self.TORSO_MASS,
            pymunk.moment_for_circle(self.TORSO_MASS, 0, 23.0),
        )
        self.torso.position = center
        self.torso_shape = pymunk.Circle(self.torso, 23.0)
        self.torso_shape.friction = 0.8
        self._set_robot_filter(self.torso_shape)
        self.space.add(self.torso, self.torso_shape)

        self.hips: list[pymunk.Body] = []
        self.shins: list[pymunk.Body] = []
        self.feet: list[pymunk.Body] = []
        self.knee_joints: list[pymunk.PivotJoint] = []
        self.hip_springs: list[pymunk.DampedRotarySpring] = []
        self.knee_springs: list[pymunk.DampedRotarySpring] = []
        self.ankle_springs: list[pymunk.DampedRotarySpring] = []
        self.knee_motors: list[pymunk.SimpleMotor] = []
        self.ankle_motors: list[pymunk.SimpleMotor] = []

        leg_angles = [
            -math.pi / 2.0,
            -math.pi / 2.0 + 2.0 * math.pi / 3.0,
            -math.pi / 2.0 + 4.0 * math.pi / 3.0,
        ]
        hip_length, shin_length, foot_length = self.limb_lengths
        for angle in leg_angles:
            hip_mass = self.HIP_MASS * hip_length / DEFAULT_LIMB_LENGTH
            shin_mass = self.SHIN_MASS * shin_length / DEFAULT_LIMB_LENGTH
            foot_mass = self.FOOT_MASS * foot_length / DEFAULT_LIMB_LENGTH
            direction = pymunk.Vec2d(math.cos(angle), math.sin(angle))
            hip_center = center + direction * (hip_length / 2.0)
            hip = pymunk.Body(
                hip_mass,
                pymunk.moment_for_segment(
                    hip_mass,
                    (-hip_length / 2.0, 0.0),
                    (hip_length / 2.0, 0.0),
                    5.0,
                ),
            )
            hip.position = hip_center
            hip.angle = angle
            hip_shape = pymunk.Segment(
                hip, (-hip_length / 2.0, 0.0), (hip_length / 2.0, 0.0), 5.0
            )
            hip_shape.friction = 0.9
            self._set_robot_filter(hip_shape)

            knee_world = hip.local_to_world((hip_length / 2.0, 0.0))
            shin = pymunk.Body(
                shin_mass,
                pymunk.moment_for_segment(
                    shin_mass,
                    (-shin_length / 2.0, 0.0),
                    (shin_length / 2.0, 0.0),
                    4.0,
                ),
            )
            shin.angle = angle + DEFAULT_KNEE_ANGLE
            shin.position = knee_world - pymunk.Vec2d(
                -shin_length / 2.0, 0.0
            ).rotated(shin.angle)
            shin_shape = pymunk.Segment(
                shin, (-shin_length / 2.0, 0.0), (shin_length / 2.0, 0.0), 4.0
            )
            shin_shape.friction = 1.0
            self._set_robot_filter(shin_shape)

            ankle_world = shin.local_to_world((shin_length / 2.0, 0.0))
            foot = pymunk.Body(
                foot_mass,
                pymunk.moment_for_segment(
                    foot_mass,
                    (-foot_length / 2.0, 0.0),
                    (foot_length / 2.0, 0.0),
                    5.0,
                ),
            )
            foot.angle = shin.angle + DEFAULT_ANKLE_ANGLE
            foot.position = ankle_world - pymunk.Vec2d(
                -foot_length / 2.0, 0.0
            ).rotated(foot.angle)
            foot_shape = pymunk.Segment(
                foot,
                (-foot_length / 2.0, 0.0),
                (foot_length / 2.0, 0.0),
                5.0,
            )
            foot_shape.friction = 1.2
            self._set_robot_filter(foot_shape)

            hip_pivot = pymunk.PivotJoint(self.torso, hip, center)
            hip_limits = pymunk.RotaryLimitJoint(
                self.torso, hip, angle - self.HIP_LIMIT, angle + self.HIP_LIMIT
            )
            hip_spring = pymunk.DampedRotarySpring(
                self.torso,
                hip,
                self.torso.angle - hip.angle,
                self.JOINT_SPRING_STIFFNESS,
                self.JOINT_SPRING_DAMPING,
            )
            hip_spring.max_force = self.JOINT_SPRING_TORQUE_LIMIT
            knee_pivot = pymunk.PivotJoint(hip, shin, knee_world)
            knee_pivot.collide_bodies = False
            knee_motor = pymunk.SimpleMotor(hip, shin, 0.0)
            knee_motor.max_force = self.MOTOR_TORQUE_LIMIT
            knee_limits = pymunk.RotaryLimitJoint(
                hip, shin, -self.KNEE_LIMIT, self.KNEE_LIMIT
            )
            knee_spring = pymunk.DampedRotarySpring(
                hip,
                shin,
                hip.angle - shin.angle,
                self.JOINT_SPRING_STIFFNESS,
                self.JOINT_SPRING_DAMPING,
            )
            knee_spring.max_force = self.JOINT_SPRING_TORQUE_LIMIT
            ankle_pivot = pymunk.PivotJoint(shin, foot, ankle_world)
            ankle_pivot.collide_bodies = False
            ankle_motor = pymunk.SimpleMotor(shin, foot, 0.0)
            ankle_motor.max_force = self.MOTOR_TORQUE_LIMIT
            ankle_limits = pymunk.RotaryLimitJoint(
                shin, foot, self.ANKLE_MIN, self.ANKLE_MAX
            )
            ankle_spring = pymunk.DampedRotarySpring(
                shin,
                foot,
                shin.angle - foot.angle,
                self.JOINT_SPRING_STIFFNESS,
                self.JOINT_SPRING_DAMPING,
            )
            ankle_spring.max_force = self.JOINT_SPRING_TORQUE_LIMIT

            self.space.add(
                hip,
                hip_shape,
                shin,
                shin_shape,
                foot,
                foot_shape,
                hip_pivot,
                hip_limits,
                hip_spring,
                knee_pivot,
                knee_motor,
                knee_limits,
                knee_spring,
                ankle_pivot,
                ankle_motor,
                ankle_limits,
                ankle_spring,
            )
            self.hip_springs.append(hip_spring)
            self.knee_springs.append(knee_spring)
            self.ankle_springs.append(ankle_spring)
            self.hips.append(hip)
            self.shins.append(shin)
            self.feet.append(foot)
            self.knee_motors.append(knee_motor)
            self.ankle_motors.append(ankle_motor)

        self.hip_initial_angle = self.hips[0].angle
        self.hip_initial_relative_angles = [
            hip.angle - self.torso.angle for hip in self.hips
        ]
        self.hip_motor = pymunk.SimpleMotor(self.torso, self.hips[0], 0.0)
        self.hip_motor.max_force = self.MOTOR_TORQUE_LIMIT
        self.space.add(self.hip_motor)

        self.hub_motor = pymunk.SimpleMotor(self.space.static_body, self.torso, 0.0)
        self.hub_motor.max_force = self.HUB_TORQUE_LIMIT
        self.space.add(self.hub_motor)
        self._motor_body_pairs: list[
            tuple[pymunk.SimpleMotor, pymunk.Body, pymunk.Body]
        ] = [
            (self.hub_motor, self.space.static_body, self.torso),
            (self.hip_motor, self.torso, self.hips[0]),
            *zip(self.knee_motors, self.hips, self.shins),
            *zip(self.ankle_motors, self.shins, self.feet),
        ]

        for index in (1, 2):
            phase = self.hips[index].angle - self.hips[0].angle
            coupling = pymunk.GearJoint(
                self.hips[0], self.hips[index], phase=phase, ratio=1.0
            )
            coupling.max_force = 1_000_000.0
            self.space.add(coupling)

    @staticmethod
    def _set_robot_filter(shape: pymunk.Shape) -> None:
        shape.filter = pymunk.ShapeFilter(group=1)

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            if self.fallen:
                self.elapsed += self.DT
                continue

            for substep_index in range(self.PHYSICS_SUBSTEPS):
                # SimpleMotor.rate is body A relative to body B, so invert B's target.
                self.hub_motor.rate = -self.genome.hub_speed
                phase = 2.0 * math.pi * self.genome.frequency * self.elapsed
                hip_target = (
                    self.genome.hip_offset
                    + self.genome.hip_amplitude * math.sin(phase)
                )
                for index, spring in enumerate(self.hip_springs):
                    spring.rest_angle = -(
                        self.hip_initial_relative_angles[index] + hip_target
                    )
                hip_relative = (
                    self.hips[0].angle - self.torso.angle - self.hip_initial_angle
                )
                self.hip_motor.rate = -9.0 * (hip_target - hip_relative)

                for index, motor in enumerate(self.knee_motors):
                    target = self.genome.knee_offset + (
                        self.genome.knee_amplitudes[index]
                        * math.sin(phase + self.genome.knee_phases[index])
                    )
                    self.knee_springs[index].rest_angle = -target
                    relative_angle = self.shins[index].angle - self.hips[index].angle
                    motor.rate = -10.0 * (target - relative_angle)

                for index, motor in enumerate(self.ankle_motors):
                    target = self.genome.ankle_offset + (
                        self.genome.ankle_amplitudes[index]
                        * math.sin(phase + self.genome.ankle_phases[index])
                    )
                    self.ankle_springs[index].rest_angle = -target
                    relative_angle = self.feet[index].angle - self.shins[index].angle
                    motor.rate = -10.0 * (target - relative_angle)

                self.space.step(self.DT / self.PHYSICS_SUBSTEPS)
                substep = self.DT / self.PHYSICS_SUBSTEPS
                self.elapsed += substep
                self.max_forward_speed = max(
                    self.max_forward_speed, self.torso.velocity.x
                )
                self.min_torso_y = min(self.min_torso_y, self.torso.position.y)
                self.max_torso_y = max(self.max_torso_y, self.torso.position.y)
                self.motor_energy += sum(
                    abs(
                        motor.impulse
                        * (body_b.angular_velocity - body_a.angular_velocity)
                    )
                    for motor, body_a, body_b in self._motor_body_pairs
                )
                if self.torso.position.y < 48.0:
                    self.fallen = True
                    self.elapsed += (
                        self.PHYSICS_SUBSTEPS - substep_index - 1
                    ) * self.DT / self.PHYSICS_SUBSTEPS
                    break

    def fitness(self, weights: FitnessWeights = DEFAULT_FITNESS_WEIGHTS) -> float:
        speed_score = self.max_forward_speed / (
            self.max_forward_speed + self.SPEED_REFERENCE
        )
        forward_speed = max(0.0, self.torso.position.x - self.start_x) / max(
            self.elapsed, self.DT
        )
        forward_distance_score = forward_speed / (
            forward_speed + self.SPEED_REFERENCE
        )
        bounce = self.max_torso_y - self.min_torso_y
        bounce_score = 1.0 / (1.0 + bounce / self.BOUNCE_REFERENCE)
        mean_power = self.motor_energy / max(self.elapsed, self.DT)
        energy_score = 1.0 / (1.0 + mean_power / self.POWER_REFERENCE)
        total_weight = (
            weights.max_speed
            + weights.forward_distance
            + weights.low_bounce
            + weights.low_energy
        )
        score = (
            weights.max_speed * speed_score
            + weights.forward_distance * forward_distance_score
            + weights.low_bounce * bounce_score
            + weights.low_energy * energy_score
        ) / total_weight
        return score - (self.FALL_PENALTY if self.fallen else 0.0)
