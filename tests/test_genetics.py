import random
import unittest

from whegwalker.genetics import (
    ANKLE_ANGLE_MAX,
    ANKLE_ANGLE_MIN,
    DEFAULT_ANKLE_ANGLE,
    DEFAULT_KNEE_ANGLE,
    DEFAULT_LIMB_LENGTH,
    GeneticAlgorithm,
    Genome,
    KNEE_ANGLE_MAX,
    KNEE_ANGLE_MIN,
    LIMB_LENGTH_MAX,
    LIMB_LENGTH_MIN,
    MAX_LIMB_LENGTH_CHANGE,
    TOTAL_LIMB_LENGTH,
)


class GenomeTests(unittest.TestCase):
    def test_random_genome_has_independent_gaits_and_shared_biases(self) -> None:
        genome = Genome.random(random.Random(7))
        self.assertEqual(len(genome.knee_amplitudes), 3)
        self.assertEqual(len(genome.knee_phases), 3)
        self.assertIsInstance(genome.knee_offset, float)
        self.assertEqual(len(genome.ankle_amplitudes), 3)
        self.assertEqual(len(genome.ankle_phases), 3)
        self.assertIsInstance(genome.ankle_offset, float)
        self.assertTrue(KNEE_ANGLE_MIN <= genome.knee_offset <= KNEE_ANGLE_MAX)
        self.assertTrue(ANKLE_ANGLE_MIN <= genome.ankle_offset <= ANKLE_ANGLE_MAX)
        self.assertEqual(
            genome.limb_lengths,
            [DEFAULT_LIMB_LENGTH] * 3,
        )

    def test_biped_genomes_evolve_two_independent_hip_drives(self) -> None:
        first = Genome.random(random.Random(70), "biped")
        second = Genome.random(random.Random(71), "biped")
        child = Genome.crossover(first, second, random.Random(72))
        child.mutate(random.Random(73), rate=1.0)
        algorithm = GeneticAlgorithm(population_size=4, robot_type="biped")

        self.assertEqual(first.robot_type, "biped")
        self.assertEqual(first.hub_speed, 0.0)
        self.assertEqual(len(first.hip_amplitudes), 2)
        self.assertEqual(len(first.hip_phases), 2)
        self.assertEqual(len(first.knee_amplitudes), 2)
        self.assertEqual(len(first.ankle_amplitudes), 2)
        self.assertEqual(child.robot_type, "biped")
        self.assertEqual(child.hub_speed, 0.0)
        self.assertEqual(len(child.hip_amplitudes), 2)
        self.assertEqual(len(algorithm.population), 4)
        self.assertTrue(all(g.robot_type == "biped" for g in algorithm.population))

    def test_mutation_respects_parameter_bounds(self) -> None:
        genome = Genome.random(random.Random(3))
        for seed in range(100):
            genome.mutate(random.Random(seed), rate=1.0)
        self.assertTrue(0.4 <= genome.frequency <= 2.8)
        self.assertTrue(-6.0 <= genome.hub_speed <= 6.0)
        self.assertTrue(0.05 <= genome.hip_amplitude <= 1.2)
        self.assertTrue(all(0.05 <= value <= 1.2 for value in genome.knee_amplitudes))
        self.assertTrue(
            all(-3.14159 <= value <= 3.14159 for value in genome.knee_phases)
        )
        self.assertTrue(KNEE_ANGLE_MIN <= genome.knee_offset <= KNEE_ANGLE_MAX)
        self.assertTrue(ANKLE_ANGLE_MIN <= genome.ankle_offset <= ANKLE_ANGLE_MAX)
        self.assertTrue(
            all(0.05 <= value <= 1.2 for value in genome.ankle_amplitudes)
        )
        self.assertTrue(
            all(-3.14159 <= value <= 3.14159 for value in genome.ankle_phases)
        )
        self.assertTrue(
            all(
                LIMB_LENGTH_MIN <= length <= LIMB_LENGTH_MAX
                for length in genome.limb_lengths
            )
        )
        self.assertEqual(len(genome.limb_lengths), 3)
        self.assertAlmostEqual(sum(genome.limb_lengths), TOTAL_LIMB_LENGTH)

    def test_length_mutation_preserves_total_and_allows_larger_changes(self) -> None:
        genome = Genome.random(random.Random(41))
        before = genome.limb_lengths.copy()

        genome.mutate(random.Random(42), rate=1.0)

        self.assertAlmostEqual(sum(genome.limb_lengths), TOTAL_LIMB_LENGTH)
        self.assertTrue(
            all(
                LIMB_LENGTH_MIN <= length <= LIMB_LENGTH_MAX
                for length in genome.limb_lengths
            )
        )
        self.assertLessEqual(
            max(abs(new - old) for old, new in zip(before, genome.limb_lengths)),
            MAX_LIMB_LENGTH_CHANGE,
        )
        self.assertGreater(
            max(abs(new - old) for old, new in zip(before, genome.limb_lengths)),
            1.5,
        )

    def test_length_mutation_respects_zero_rate(self) -> None:
        genome = Genome.random(random.Random(51))
        before = genome.limb_lengths.copy()

        genome.mutate(random.Random(52), rate=0.0)

        self.assertEqual(genome.limb_lengths, before)

    def test_crossover_keeps_a_complete_length_profile(self) -> None:
        first = Genome.random(random.Random(61))
        first.limb_lengths = [35.0, 43.0, 60.0]
        second = Genome.random(random.Random(62))
        second.limb_lengths = [55.0, 50.0, 33.0]

        child = Genome.crossover(first, second, random.Random(63))

        self.assertIn(child.limb_lengths, [first.limb_lengths, second.limb_lengths])
        self.assertAlmostEqual(sum(child.limb_lengths), TOTAL_LIMB_LENGTH)

    def test_mutation_can_move_joints_far_from_default_bias(self) -> None:
        knee_deviations = []
        ankle_deviations = []
        for seed in range(30):
            genome = Genome.random(random.Random(seed + 11))
            genome.knee_offset = DEFAULT_KNEE_ANGLE
            genome.ankle_offset = DEFAULT_ANKLE_ANGLE
            genome.mutate(random.Random(seed), rate=1.0)
            knee_deviations.append(abs(genome.knee_offset - DEFAULT_KNEE_ANGLE))
            ankle_deviations.append(abs(genome.ankle_offset - DEFAULT_ANKLE_ANGLE))
        self.assertGreater(max(knee_deviations), 0.35)
        self.assertGreater(max(ankle_deviations), 0.35)

    def test_angle_mutation_has_a_small_bias_towards_neutral(self) -> None:
        knee_start = 1.0
        ankle_start = -0.1
        knee_mutations = []
        ankle_mutations = []
        for seed in range(1000):
            genome = Genome.random(random.Random(seed))
            genome.knee_offset = knee_start
            genome.ankle_offset = ankle_start
            genome.mutate(random.Random(seed + 1000), rate=1.0)
            knee_mutations.append(genome.knee_offset - knee_start)
            ankle_mutations.append(genome.ankle_offset - ankle_start)

        mean_knee_change = sum(knee_mutations) / len(knee_mutations)
        mean_ankle_change = sum(ankle_mutations) / len(ankle_mutations)
        self.assertLess(mean_knee_change, -0.04)
        self.assertGreater(mean_ankle_change, 0.02)
        self.assertGreater(max(knee_mutations), 0.3)
        self.assertLess(min(knee_mutations), -0.3)

    def test_zero_mutation_rate_keeps_joint_offsets_unchanged(self) -> None:
        genome = Genome.random(random.Random(11))
        knee_offset = genome.knee_offset
        ankle_offset = genome.ankle_offset

        genome.mutate(random.Random(12), rate=0.0)

        self.assertEqual(genome.knee_offset, knee_offset)
        self.assertEqual(genome.ankle_offset, ankle_offset)

    def test_crossover_inherits_each_shared_joint_bias_as_a_scalar(self) -> None:
        first = Genome.random(random.Random(17))
        first.knee_offset = -0.8
        first.ankle_offset = 0.9
        second = Genome.random(random.Random(18))
        second.knee_offset = 0.7
        second.ankle_offset = 0.2

        child = Genome.crossover(first, second, random.Random(19))

        self.assertIn(child.knee_offset, [first.knee_offset, second.knee_offset])
        self.assertIn(child.ankle_offset, [first.ankle_offset, second.ankle_offset])

    def test_generation_keeps_elites_and_updates_best(self) -> None:
        algorithm = GeneticAlgorithm(population_size=4, elite_count=1, seed=2)
        original_best = algorithm.population[2].copy()
        algorithm.next_generation([1.0, 2.0, 9.0, 3.0])
        self.assertEqual(algorithm.generation, 2)
        self.assertEqual(algorithm.best_fitness, 9.0)
        self.assertEqual(algorithm.best_genome, original_best)
        self.assertEqual(len(algorithm.population), 4)

    def test_invalid_fitness_count_is_rejected(self) -> None:
        algorithm = GeneticAlgorithm(population_size=4)
        with self.assertRaises(ValueError):
            algorithm.next_generation([1.0])


if __name__ == "__main__":
    unittest.main()
