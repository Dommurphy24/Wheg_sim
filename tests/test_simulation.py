import math
import random
import time
import unittest

from whegwalker.genetics import (
    DEFAULT_ANKLE_ANGLE,
    DEFAULT_KNEE_ANGLE,
    DEFAULT_LIMB_LENGTH,
    Genome,
)
from whegwalker.app import (
    FITNESS_SLIDER_LEFT,
    FITNESS_SLIDER_RIGHT,
    FITNESS_SLIDER_SPACING,
    FITNESS_SLIDER_TOP,
    HEIGHT,
    SimulatorApp,
    WIDTH,
    _canvas_viewport,
    _fit_window_size,
    _window_to_canvas,
)
from whegwalker.physics import FitnessWeights, WhegSimulation
from whegwalker.trainer import EvolutionWorker, Trainer


def test_genome() -> Genome:
    return Genome(
        frequency=1.0,
        hub_speed=4.0,
        hip_amplitude=0.5,
        hip_offset=0.0,
        knee_amplitudes=[0.35, 0.6, 0.8],
        knee_phases=[0.0, math.pi / 2.0, math.pi],
        knee_offset=0.0,
        ankle_amplitudes=[0.25, 0.5, 0.75],
        ankle_phases=[math.pi / 2.0, math.pi, 0.0],
        ankle_offset=0.0,
        limb_lengths=[DEFAULT_LIMB_LENGTH] * 3,
    )


class SimulationTests(unittest.TestCase):
    def test_initial_window_fits_desktop_and_preserves_aspect_ratio(self) -> None:
        window_width, window_height = _fit_window_size((1024, 768))

        self.assertLessEqual(window_width, 1024 * 0.9 + 1)
        self.assertLessEqual(window_height, 768 * 0.9 + 1)
        self.assertAlmostEqual(window_width / window_height, WIDTH / HEIGHT, places=2)

    def test_mouse_coordinates_map_through_scaled_window_viewport(self) -> None:
        window_size = _fit_window_size((1024, 768))
        viewport_x, viewport_y, viewport_width, viewport_height = _canvas_viewport(
            window_size
        )
        canvas_position = (FITNESS_SLIDER_LEFT, 658)
        window_position = (
            viewport_x + round(canvas_position[0] * viewport_width / WIDTH),
            viewport_y + round(canvas_position[1] * viewport_height / HEIGHT),
        )
        mapped_position = _window_to_canvas(window_position, window_size)

        self.assertAlmostEqual(mapped_position[0], canvas_position[0], delta=2)
        self.assertAlmostEqual(mapped_position[1], canvas_position[1], delta=2)

    def test_fitness_weights_combine_normalized_speed_bounce_and_energy(self) -> None:
        simulation = WhegSimulation(test_genome())
        simulation.elapsed = 5.0
        simulation.torso.position = (simulation.start_x + 1000.0, 160.0)
        simulation.max_forward_speed = simulation.SPEED_REFERENCE
        simulation.min_torso_y = 120.0
        simulation.max_torso_y = 200.0
        simulation.motor_energy = simulation.POWER_REFERENCE * simulation.elapsed

        self.assertAlmostEqual(simulation.fitness(), 0.5)
        self.assertAlmostEqual(
            simulation.fitness(
                FitnessWeights(
                    max_speed=100.0,
                    forward_distance=0.0,
                    low_bounce=0.0,
                    low_energy=0.0,
                )
            ),
            0.5,
        )
        self.assertAlmostEqual(
            simulation.fitness(
                FitnessWeights(
                    max_speed=0.0,
                    forward_distance=0.0,
                    low_bounce=100.0,
                    low_energy=0.0,
                )
            ),
            0.5,
        )
        self.assertAlmostEqual(
            simulation.fitness(
                FitnessWeights(
                    max_speed=0.0,
                    forward_distance=0.0,
                    low_bounce=0.0,
                    low_energy=100.0,
                )
            ),
            0.5,
        )

    def test_falling_applies_a_full_point_fitness_penalty(self) -> None:
        standing = WhegSimulation(test_genome())
        fallen = WhegSimulation(test_genome())
        for simulation in (standing, fallen):
            simulation.elapsed = 5.0
            simulation.torso.position = (
                simulation.start_x + 500.0,
                simulation.torso.position.y,
            )
            simulation.max_forward_speed = 100.0
            simulation.min_torso_y = 140.0
            simulation.max_torso_y = 180.0
            simulation.motor_energy = 1_000_000.0
        fallen.fallen = True

        self.assertAlmostEqual(
            standing.fitness() - fallen.fitness(),
            WhegSimulation.FALL_PENALTY,
        )
        self.assertLess(fallen.fitness(), 0.0)

    def test_fall_penalty_weight_scales_the_fall_penalty_only(self) -> None:
        weights = FitnessWeights(fall_penalty=35.0)
        standing = WhegSimulation(test_genome())
        fallen = WhegSimulation(test_genome())
        for simulation in (standing, fallen):
            simulation.elapsed = 5.0
            simulation.torso.position = (simulation.start_x + 500.0, 160.0)
            simulation.max_forward_speed = 100.0
            simulation.min_torso_y = 140.0
            simulation.max_torso_y = 180.0
            simulation.motor_energy = 1_000_000.0
        fallen.fallen = True

        self.assertAlmostEqual(
            standing.fitness(weights) - fallen.fitness(weights),
            WhegSimulation.FALL_PENALTY * 0.35,
        )

    def test_fall_penalty_halves_each_half_life_upright(self) -> None:
        simulations = [
            WhegSimulation(test_genome())
            for _ in range(3)
        ]
        for simulation, upright_time in zip(
            simulations,
            (
                0.0,
                WhegSimulation.FALL_PENALTY_HALF_LIFE,
                2.0 * WhegSimulation.FALL_PENALTY_HALF_LIFE,
            ),
        ):
            simulation.fallen = True
            simulation.upright_time = upright_time

        base_penalty = WhegSimulation.FALL_PENALTY
        scores = [simulation.fitness() for simulation in simulations]
        self.assertAlmostEqual(scores[1] - scores[0], base_penalty * 0.5)
        self.assertAlmostEqual(scores[2] - scores[1], base_penalty * 0.25)

    def test_distance_per_time_reward_distinguishes_forward_progress(self) -> None:
        weights = FitnessWeights(
            max_speed=0.0,
            forward_distance=100.0,
            low_bounce=0.0,
            low_energy=0.0,
        )
        stationary = WhegSimulation(test_genome())
        moving = WhegSimulation(test_genome())
        for simulation in (stationary, moving):
            simulation.elapsed = 5.0
        moving.torso.position = (
            moving.start_x + WhegSimulation.SPEED_REFERENCE * 5.0,
            moving.torso.position.y,
        )

        self.assertGreater(moving.fitness(weights), stationary.fitness(weights))
        self.assertAlmostEqual(moving.fitness(weights), 0.5)

    def test_fitness_weights_reject_invalid_or_all_zero_values(self) -> None:
        with self.assertRaises(ValueError):
            FitnessWeights(
                max_speed=0.0,
                forward_distance=0.0,
                low_bounce=0.0,
                low_energy=0.0,
            )
        with self.assertRaises(ValueError):
            FitnessWeights(max_speed=101.0)
        with self.assertRaises(ValueError):
            FitnessWeights(
                max_speed=0.0,
                forward_distance=0.0,
                low_bounce=0.0,
                low_energy=0.0,
                fall_penalty=0.0,
            )

    def test_trial_records_speed_bounce_and_energy_metrics(self) -> None:
        simulation = WhegSimulation(test_genome())
        simulation.step(120)

        self.assertGreater(simulation.max_forward_speed, 0.0)
        self.assertGreater(simulation.max_torso_y, simulation.min_torso_y)
        self.assertGreater(simulation.motor_energy, 0.0)

    def test_small_random_disturbances_are_seedable_and_affect_motion(self) -> None:
        first = WhegSimulation(
            test_genome(), noise_seed=123, terrain_seed=456
        )
        repeat = WhegSimulation(
            test_genome(), noise_seed=123, terrain_seed=456
        )
        different = WhegSimulation(
            test_genome(), noise_seed=124, terrain_seed=456
        )

        for simulation in (first, repeat, different):
            simulation.step(120)

        self.assertAlmostEqual(
            (first.torso.position - repeat.torso.position).length, 0.0, places=10
        )
        self.assertGreater(
            (first.torso.position - different.torso.position).length, 0.001
        )

    def test_seeded_terrain_is_smooth_random_and_scanned_ahead(self) -> None:
        first = WhegSimulation(test_genome(), terrain_seed=321)
        repeat = WhegSimulation(test_genome(), terrain_seed=321)
        different = WhegSimulation(test_genome(), terrain_seed=322)
        sample_xs = [
            first.start_x + offset for offset in range(0, 700, 24)
        ]
        first_profile = [first.terrain_height_at(x) for x in sample_xs]
        repeat_profile = [repeat.terrain_height_at(x) for x in sample_xs]
        different_profile = [different.terrain_height_at(x) for x in sample_xs]

        self.assertEqual(first_profile, repeat_profile)
        self.assertNotEqual(first_profile, different_profile)
        self.assertGreater(max(first_profile) - min(first_profile), 1.0)
        self.assertEqual(
            first.terrain_sensor_range, 2.0 * first.robot_height
        )
        self.assertEqual(
            len(first.terrain_sensor_readings),
            first.TERRAIN_SENSOR_SAMPLE_COUNT,
        )
        expected_readings = tuple(
            first.terrain_height_at(first.start_x + distance)
            - first.ground_y
            - first.TERRAIN_RADIUS
            for distance in first.terrain_sensor_distances
        )
        self.assertEqual(first.terrain_sensor_readings, expected_readings)

    def test_forward_terrain_sensor_increases_knee_clearance_target(self) -> None:
        simulation = WhegSimulation(
            Genome.random(random.Random(83), "biped"), terrain_seed=84
        )
        flat_readings = (0.0,) * simulation.TERRAIN_SENSOR_SAMPLE_COUNT
        simulation.terrain_sensor_readings = flat_readings
        self.assertEqual(simulation._terrain_knee_lift(), 0.0)

        simulation.terrain_sensor_readings = (
            0.0,
            2.0,
            5.0,
            10.0,
            8.0,
            6.0,
            4.0,
            2.0,
            0.0,
        )
        self.assertGreater(simulation._terrain_knee_lift(), 0.0)

    def test_fitness_sliders_update_weights_and_prevent_zero_total(self) -> None:
        app = SimulatorApp.__new__(SimulatorApp)
        app.fitness_weights = FitnessWeights()

        self.assertEqual(
            app._fitness_slider_at(
                (FITNESS_SLIDER_LEFT, FITNESS_SLIDER_TOP)
            ),
            "max_speed",
        )
        self.assertEqual(
            app._fitness_slider_at(
                (
                    FITNESS_SLIDER_RIGHT,
                    FITNESS_SLIDER_TOP + 4 * FITNESS_SLIDER_SPACING,
                )
            ),
            "fall_penalty",
        )
        app._set_fitness_slider("max_speed", FITNESS_SLIDER_LEFT)
        app._set_fitness_slider("forward_distance", FITNESS_SLIDER_LEFT)
        app._set_fitness_slider("low_bounce", FITNESS_SLIDER_LEFT)
        app._set_fitness_slider("low_energy", FITNESS_SLIDER_LEFT)

        self.assertEqual(app.fitness_weights.max_speed, 0)
        self.assertEqual(app.fitness_weights.forward_distance, 0)
        self.assertEqual(app.fitness_weights.low_bounce, 0)
        self.assertEqual(app.fitness_weights.low_energy, 1)

        app._set_fitness_slider("fall_penalty", FITNESS_SLIDER_LEFT)
        self.assertEqual(app.fitness_weights.fall_penalty, 0)

    def test_robot_selector_changes_mode_and_restarts_evolution(self) -> None:
        app = SimulatorApp.__new__(SimulatorApp)
        app.robot_type = "wheg"
        restart_calls = []
        app._restart_evolution = lambda: restart_calls.append(True)

        self.assertEqual(app._robot_button_at((900, 560)), "wheg")
        self.assertEqual(app._robot_button_at((970, 560)), "biped")
        self.assertIsNone(app._robot_button_at((900, 600)))
        app._set_robot_type("biped")

        self.assertEqual(app.robot_type, "biped")
        self.assertEqual(restart_calls, [True])

    def test_all_leg_segments_have_equal_length_and_connected_anchors(self) -> None:
        genome = test_genome()
        genome.limb_lengths = [34.0, 45.0, 59.0]
        simulation = WhegSimulation(genome)
        self.assertEqual(simulation.limb_lengths, genome.limb_lengths)
        for hip, shin, foot in zip(
            simulation.hips, simulation.shins, simulation.feet
        ):
            lengths = simulation.limb_lengths
            hip_end = hip.local_to_world((lengths[0] / 2.0, 0.0))
            shin_start = shin.local_to_world((-lengths[1] / 2.0, 0.0))
            shin_end = shin.local_to_world((lengths[1] / 2.0, 0.0))
            foot_start = foot.local_to_world((-lengths[2] / 2.0, 0.0))
            self.assertLess((hip_end - shin_start).length, 1e-6)
            self.assertLess((shin_end - foot_start).length, 1e-6)
            self.assertAlmostEqual(
                (hip.local_to_world((lengths[0] / 2.0, 0.0))
                 - hip.local_to_world((-lengths[0] / 2.0, 0.0))).length,
                lengths[0],
            )

    def test_biped_has_two_three_dof_legs_and_no_wheg_hub_drive(self) -> None:
        genome = Genome.random(random.Random(81), "biped")
        simulation = WhegSimulation(genome)
        simulation.step(2)

        self.assertEqual(simulation.robot_type, "biped")
        self.assertEqual(simulation.leg_count, 2)
        self.assertEqual(len(simulation.hips), 2)
        self.assertEqual(len(simulation.shins), 2)
        self.assertEqual(len(simulation.feet), 2)
        self.assertEqual(len(simulation.hip_motors), 2)
        self.assertEqual(len(simulation.knee_motors), 2)
        self.assertEqual(len(simulation.ankle_motors), 2)
        self.assertTrue(
            all(
                motor.max_force == simulation.MOTOR_TORQUE_LIMIT
                for motor in simulation.hip_motors
            )
        )
        self.assertEqual(simulation.hub_motor.max_force, 0.0)
        self.assertEqual(simulation.genome.hub_speed, 0.0)
        self.assertTrue(any(abs(motor.rate) > 0.0 for motor in simulation.hip_motors))
        self.assertEqual(len(simulation.hip_anchors), 2)
        self.assertAlmostEqual(
            (simulation.hip_anchors[0] - simulation.hip_anchors[1]).length, 0.0
        )

    def test_biped_trial_ends_when_torso_capsule_touches_ground(self) -> None:
        simulation = WhegSimulation(Genome.random(random.Random(82), "biped"))
        simulation.space.gravity = (0.0, 0.0)
        simulation.torso.position = (simulation.start_x, 70.0)

        simulation.step()
        self.assertFalse(simulation.fallen)

        simulation.torso.position = (simulation.start_x, 60.0)
        simulation.step()

        self.assertTrue(simulation.fallen)
        self.assertGreater(simulation.torso.position.y, 48.0)

    def test_motor_torque_is_bounded_to_prevent_velocity_runaway(self) -> None:
        simulation = WhegSimulation(test_genome())
        motor_limits = [
            simulation.hub_motor.max_force,
            simulation.hip_motor.max_force,
            *(motor.max_force for motor in simulation.knee_motors),
            *(motor.max_force for motor in simulation.ankle_motors),
        ]

        self.assertTrue(
            all(
                0.0 < limit == simulation.MOTOR_TORQUE_LIMIT
                for limit in motor_limits[1:]
            )
        )
        self.assertEqual(motor_limits[0], simulation.HUB_TORQUE_LIMIT)

    def test_joint_motor_rate_is_not_clamped(self) -> None:
        genome = test_genome()
        genome.hub_speed = 0.0
        genome.hip_amplitude = 0.0
        genome.knee_offset = 1.2
        genome.knee_amplitudes = [0.0, 0.0, 0.0]
        genome.ankle_amplitudes = [0.0, 0.0, 0.0]
        simulation = WhegSimulation(genome)
        expected_rate = -10.0 * (
            genome.knee_offset
            - (simulation.shins[0].angle - simulation.hips[0].angle)
        )

        simulation.step()

        self.assertGreater(abs(expected_rate), 8.0)
        self.assertAlmostEqual(simulation.knee_motors[0].rate, expected_rate, delta=1.0)

    def test_unpowered_joint_yields_to_impact_and_springs_back(self) -> None:
        genome = test_genome()
        genome.hub_speed = 0.0
        genome.hip_amplitude = 0.0
        genome.knee_offset = DEFAULT_KNEE_ANGLE
        genome.knee_amplitudes = [0.0, 0.0, 0.0]
        genome.ankle_offset = DEFAULT_ANKLE_ANGLE
        genome.ankle_amplitudes = [0.0, 0.0, 0.0]
        simulation = WhegSimulation(genome, noise_seed=1, terrain_seed=2)
        simulation.space.gravity = (0.0, 0.0)
        motors = [
            simulation.hub_motor,
            simulation.hip_motor,
            *simulation.knee_motors,
            *simulation.ankle_motors,
        ]
        for motor in motors:
            motor.max_force = 0.0

        initial_knee_angle = (
            simulation.shins[0].angle - simulation.hips[0].angle
        )
        simulation.shins[0].angular_velocity += 30.0
        peak_deflection = 0.0
        for _ in range(600):
            simulation.step()
            knee_angle = (
                simulation.shins[0].angle - simulation.hips[0].angle
            )
            peak_deflection = max(
                peak_deflection, abs(knee_angle - initial_knee_angle)
            )

        final_deflection = abs(
            simulation.shins[0].angle
            - simulation.hips[0].angle
            - initial_knee_angle
        )
        self.assertGreater(peak_deflection, math.radians(15.0))
        self.assertLess(final_deflection, peak_deflection * 0.6)
        self.assertFalse(simulation.fallen)

    def test_knee_motor_tracks_target_under_gravity_with_torque_reserve(self) -> None:
        genome = Genome(
            frequency=1.0,
            hub_speed=0.0,
            hip_amplitude=0.0,
            hip_offset=0.0,
            knee_amplitudes=[0.0, 0.0, 0.0],
            knee_phases=[0.0, 0.0, 0.0],
            knee_offset=0.25,
            ankle_amplitudes=[0.0, 0.0, 0.0],
            ankle_phases=[0.0, 0.0, 0.0],
            ankle_offset=DEFAULT_ANKLE_ANGLE,
            limb_lengths=[DEFAULT_LIMB_LENGTH] * 3,
        )
        simulation = WhegSimulation(genome, noise_seed=0, terrain_seed=0)
        bodies = [
            simulation.torso,
            *simulation.hips,
            *simulation.shins,
            *simulation.feet,
        ]
        for body in bodies:
            body.position += (0.0, 5000.0)

        substep = simulation.DT / simulation.PHYSICS_SUBSTEPS
        peak_torque_fraction = 0.0
        for _ in range(30):
            simulation.step()
            peak_torque_fraction = max(
                peak_torque_fraction,
                *(
                    abs(motor.impulse)
                    / (simulation.MOTOR_TORQUE_LIMIT * substep)
                    for motor in simulation.knee_motors
                ),
            )

        knee_errors = [
            abs(shin.angle - hip.angle - genome.knee_offset)
            for hip, shin in zip(simulation.hips, simulation.shins)
        ]
        self.assertLess(max(knee_errors), math.radians(6.0))
        self.assertLess(peak_torque_fraction, 0.5)
        self.assertFalse(simulation.fallen)

    def test_initial_knee_and_ankle_angles_use_requested_defaults(self) -> None:
        simulation = WhegSimulation(test_genome())
        knee_angles = [
            shin.angle - hip.angle
            for hip, shin in zip(simulation.hips, simulation.shins)
        ]
        ankle_angles = [
            foot.angle - shin.angle
            for shin, foot in zip(simulation.shins, simulation.feet)
        ]

        for angle in knee_angles:
            self.assertAlmostEqual(angle, DEFAULT_KNEE_ANGLE)
        for angle in ankle_angles:
            self.assertAlmostEqual(angle, DEFAULT_ANKLE_ANGLE)

    def test_fast_downward_motion_does_not_tunnel_through_ground(self) -> None:
        simulation = WhegSimulation(test_genome())
        simulation.space.gravity = (0.0, 0.0)
        simulation.torso.position = (simulation.start_x, 60.0)
        simulation.torso.velocity = (0.0, -2000.0)

        simulation.step()

        self.assertGreaterEqual(
            simulation.torso.position.y,
            simulation.ground_y + simulation.ground.radius,
        )

    def test_high_energy_gait_keeps_torso_above_ground(self) -> None:
        generator = random.Random(222)
        genome = Genome.random(generator)
        for _ in range(10):
            genome = Genome.random(generator)
        simulation = WhegSimulation(genome)

        lowest_torso_y = simulation.torso.position.y
        max_body_speed = 0.0
        for _ in range(600):
            simulation.step()
            lowest_torso_y = min(lowest_torso_y, simulation.torso.position.y)
            bodies = [
                simulation.torso,
                *simulation.hips,
                *simulation.shins,
                *simulation.feet,
            ]
            max_body_speed = max(
                max_body_speed, *(body.velocity.length for body in bodies)
            )

        self.assertGreater(
            lowest_torso_y,
            simulation.ground_y + simulation.ground.radius + 23.0 - 5.0,
        )
        self.assertLess(max_body_speed, 2_500.0)
        self.assertFalse(simulation.fallen)

    def test_ground_covers_robot_movement_range(self) -> None:
        simulation = WhegSimulation(test_genome(), start_x=-5000.0)

        self.assertLess(simulation.terrain_start_x, simulation.start_x)
        self.assertGreater(simulation.terrain_end_x, simulation.start_x)
        self.assertGreater(
            simulation.terrain_end_x,
            simulation.start_x + simulation.terrain_sensor_range,
        )

    def test_positive_hub_command_rotates_hub_forward(self) -> None:
        genome = Genome(
            frequency=1.0,
            hub_speed=3.0,
            hip_amplitude=0.0,
            hip_offset=0.0,
            knee_amplitudes=[0.0, 0.0, 0.0],
            knee_phases=[0.0, 0.0, 0.0],
            knee_offset=DEFAULT_KNEE_ANGLE,
            ankle_amplitudes=[0.0, 0.0, 0.0],
            ankle_phases=[0.0, 0.0, 0.0],
            ankle_offset=DEFAULT_ANKLE_ANGLE,
            limb_lengths=[DEFAULT_LIMB_LENGTH] * 3,
        )
        simulation = WhegSimulation(genome)
        simulation.space.gravity = (0.0, 0.0)
        for body in [
            simulation.torso,
            *simulation.hips,
            *simulation.shins,
            *simulation.feet,
        ]:
            body.position += (0.0, 400.0)
            body.velocity = (0.0, 0.0)

        simulation.step(8)

        self.assertGreater(simulation.torso.angular_velocity, 0.0)

    def test_hub_motor_overcomes_gravity_without_saturating(self) -> None:
        genome = test_genome()
        genome.hub_speed = 3.0
        genome.hip_amplitude = 0.0
        genome.knee_amplitudes = [1.0, 1.0, 1.0]
        genome.knee_phases = [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
        genome.knee_offset = DEFAULT_KNEE_ANGLE
        genome.ankle_amplitudes = [0.6, 0.6, 0.6]
        genome.ankle_phases = [
            math.pi / 3.0,
            math.pi,
            5.0 * math.pi / 3.0,
        ]
        genome.ankle_offset = DEFAULT_ANKLE_ANGLE
        simulation = WhegSimulation(genome)
        bodies = [
            simulation.torso,
            *simulation.hips,
            *simulation.shins,
            *simulation.feet,
        ]
        for body in bodies:
            body.position += (0.0, 5000.0)

        initial_angle = simulation.torso.angle
        peak_torque_fraction = 0.0
        substep = simulation.DT / simulation.PHYSICS_SUBSTEPS
        for _ in range(240):
            simulation.step()
            peak_torque_fraction = max(
                peak_torque_fraction,
                abs(simulation.hub_motor.impulse)
                / (simulation.HUB_TORQUE_LIMIT * substep),
            )

        self.assertGreater(simulation.torso.angle - initial_angle, math.pi)
        self.assertLess(peak_torque_fraction, 0.5)
        self.assertFalse(simulation.fallen)

    def test_hub_and_joints_actuate_without_marking_hub_rotation_as_fall(self) -> None:
        simulation = WhegSimulation(test_genome())
        initial_hip_angles = [
            hip.angle - simulation.torso.angle for hip in simulation.hips
        ]
        initial_knee_angles = [
            shin.angle - hip.angle
            for hip, shin in zip(simulation.hips, simulation.shins)
        ]

        peak_hub_speed = 0.0
        for _ in range(120):
            simulation.step()
            peak_hub_speed = max(
                peak_hub_speed, abs(simulation.torso.angular_velocity)
            )

        self.assertGreater(peak_hub_speed, 1.0)
        self.assertGreater(abs(simulation.torso.angle), 0.1)
        simulation.torso.angle = 1.5
        simulation.step()
        self.assertFalse(simulation.fallen)
        self.assertTrue(
            any(
                abs(hip.angle - simulation.torso.angle - initial) > 0.05
                for hip, initial in zip(simulation.hips, initial_hip_angles)
            )
        )
        self.assertTrue(
            any(
                abs(shin.angle - hip.angle - initial) > 0.05
                for hip, shin, initial in zip(
                    simulation.hips, simulation.shins, initial_knee_angles
                )
            )
        )

    def test_knee_motors_follow_different_randomized_targets(self) -> None:
        genome = Genome(
            frequency=1.0,
            hub_speed=0.0,
            hip_amplitude=0.0,
            hip_offset=0.0,
            knee_amplitudes=[0.6, 0.6, 0.6],
            knee_phases=[-math.pi / 2.0, 0.0, math.pi / 2.0],
            knee_offset=DEFAULT_KNEE_ANGLE,
            ankle_amplitudes=[0.0, 0.0, 0.0],
            ankle_phases=[0.0, 0.0, 0.0],
            ankle_offset=DEFAULT_ANKLE_ANGLE,
            limb_lengths=[DEFAULT_LIMB_LENGTH] * 3,
        )
        simulation = WhegSimulation(genome)
        simulation.space.gravity = (0.0, 0.0)

        simulation.step(30)

        knee_angles = [
            shin.angle - hip.angle
            for hip, shin in zip(simulation.hips, simulation.shins)
        ]
        self.assertLess(knee_angles[0], DEFAULT_KNEE_ANGLE)
        self.assertGreater(knee_angles[2], DEFAULT_KNEE_ANGLE)
        self.assertLess(abs(simulation.torso.angle), 0.01)

    def test_ankle_motors_follow_independent_targets(self) -> None:
        genome = test_genome()
        genome.hub_speed = 0.0
        genome.hip_amplitude = 0.0
        genome.knee_offset = DEFAULT_KNEE_ANGLE
        genome.knee_amplitudes = [0.0, 0.0, 0.0]
        genome.ankle_amplitudes = [0.6, 0.6, 0.6]
        genome.ankle_phases = [-math.pi / 2.0, 0.0, math.pi / 2.0]
        genome.ankle_offset = DEFAULT_ANKLE_ANGLE
        simulation = WhegSimulation(genome)
        simulation.space.gravity = (0.0, 0.0)

        simulation.step(30)

        ankle_angles = [
            foot.angle - shin.angle
            for shin, foot in zip(simulation.shins, simulation.feet)
        ]
        self.assertLess(ankle_angles[0], DEFAULT_ANKLE_ANGLE)
        self.assertGreater(ankle_angles[2], DEFAULT_ANKLE_ANGLE)
        self.assertEqual(len(simulation.feet), 3)

    def test_joint_bias_targets_are_shared_across_legs(self) -> None:
        genome = test_genome()
        genome.hub_speed = 0.0
        genome.hip_amplitude = 0.0
        genome.knee_amplitudes = [0.0, 0.0, 0.0]
        genome.ankle_amplitudes = [0.0, 0.0, 0.0]
        genome.knee_offset = DEFAULT_KNEE_ANGLE + 0.2
        genome.ankle_offset = DEFAULT_ANKLE_ANGLE - 0.2
        simulation = WhegSimulation(genome)
        simulation.space.gravity = (0.0, 0.0)

        simulation.step()

        for motors in (simulation.knee_motors, simulation.ankle_motors):
            for motor in motors[1:]:
                self.assertAlmostEqual(motor.rate, motors[0].rate, delta=0.2)

    def test_three_hips_remain_rigidly_coupled(self) -> None:
        simulation = WhegSimulation(test_genome())
        initial_differences = [
            simulation.hips[index].angle - simulation.hips[0].angle
            for index in (1, 2)
        ]

        simulation.step(240)

        final_differences = [
            simulation.hips[index].angle - simulation.hips[0].angle
            for index in (1, 2)
        ]
        for initial, final in zip(initial_differences, final_differences):
            self.assertAlmostEqual(initial, final, delta=0.08)
        self.assertEqual(len(simulation.knee_motors), 3)
        self.assertEqual(len(simulation.ankle_motors), 3)

    def test_training_evaluates_population_and_advances_generation(self) -> None:
        weights = FitnessWeights(
            max_speed=100.0,
            forward_distance=0.0,
            low_bounce=0.0,
            low_energy=0.0,
        )
        trainer = Trainer(population_size=2, fitness_weights=weights)
        first_candidate_terrain = [
            trainer.simulation.terrain_height_at(trainer.simulation.start_x + offset)
            for offset in range(0, 500, 48)
        ]
        trainer.update(600)
        self.assertEqual(trainer.generation, 1)
        self.assertEqual(trainer.evaluating, 2)
        self.assertEqual(
            first_candidate_terrain,
            [
                trainer.simulation.terrain_height_at(
                    trainer.simulation.start_x + offset
                )
                for offset in range(0, 500, 48)
            ],
        )

        first_generation_seed = trainer.terrain_seed
        trainer.update(600)

        self.assertEqual(trainer.generation, 2)
        self.assertEqual(trainer.evaluating, 1)
        self.assertEqual(len(trainer.history), 1)
        self.assertNotEqual(trainer.terrain_seed, first_generation_seed)

    def test_trainer_reset_preserves_selected_robot_type(self) -> None:
        trainer = Trainer(population_size=2, robot_type="biped")
        trainer.update(5)

        trainer.reset()

        self.assertEqual(trainer.robot_type, "biped")
        self.assertTrue(
            all(genome.robot_type == "biped" for genome in trainer.algorithm.population)
        )
        self.assertEqual(trainer.simulation.robot_type, "biped")
        self.assertEqual(trainer.candidate_index, 0)


class EvolutionWorkerTests(unittest.TestCase):
    def test_worker_evaluates_candidates_independently(self) -> None:
        worker = EvolutionWorker(
            population_size=2,
            trial_seconds=0.1,
            robot_type="biped",
            fitness_weights=FitnessWeights(
                max_speed=100.0,
                forward_distance=0.0,
                low_bounce=0.0,
                low_energy=0.0,
            ),
        )
        worker.start()
        try:
            deadline = time.monotonic() + 15.0
            status = worker.poll()
            while status["best_fitness"] is None and status["error"] is None:
                if time.monotonic() >= deadline:
                    self.fail("evolution worker did not finish its first candidate")
                time.sleep(0.05)
                status = worker.poll()

            self.assertIsNone(status["error"])
            self.assertGreaterEqual(status["generation"], 1)
            self.assertGreaterEqual(status["evaluated"], 0)
            self.assertLessEqual(status["evaluated"], status["population_size"])
            self.assertEqual(status["population_size"], 2)
            self.assertIsNotNone(status["best_genome"])
            self.assertEqual(status["best_genome"].robot_type, "biped")
            self.assertTrue(worker.running)
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main()
