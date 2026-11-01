from __future__ import annotations

import multiprocessing
import queue
import time

from whegwalker.genetics import GeneticAlgorithm
from whegwalker.physics import WhegSimulation


def _evolution_worker(
    population_size: int,
    trial_seconds: float,
    messages: multiprocessing.queues.Queue,
    stop_event: multiprocessing.synchronize.Event,
    pause_event: multiprocessing.synchronize.Event,
) -> None:
    algorithm = GeneticAlgorithm(population_size=population_size)
    best_fitness = float("-inf")
    best_genome = algorithm.population[0].copy()
    history: list[float] = []
    steps_per_trial = round(trial_seconds / WhegSimulation.DT)

    def publish(
        generation: int,
        evaluated: int,
        generation_best: float | None = None,
    ) -> None:
        messages.put(
            {
                "generation": generation,
                "evaluated": evaluated,
                "population_size": population_size,
                "best_fitness": best_fitness if best_fitness != float("-inf") else None,
                "generation_best": generation_best,
                "history": history.copy(),
                "best_genome": best_genome.copy(),
            }
        )

    try:
        publish(algorithm.generation, 0)
        while not stop_event.is_set():
            generation_scores: list[float] = []
            for candidate_index, genome in enumerate(algorithm.population):
                if stop_event.is_set():
                    return

                simulation = WhegSimulation(genome)
                for step_index in range(0, steps_per_trial, 30):
                    if stop_event.is_set():
                        return
                    while pause_event.is_set():
                        if stop_event.is_set():
                            return
                        time.sleep(0.05)
                    simulation.step(min(30, steps_per_trial - step_index))

                score = simulation.fitness()
                generation_scores.append(score)
                if score > best_fitness:
                    best_fitness = score
                    best_genome = genome.copy()
                publish(
                    algorithm.generation,
                    candidate_index + 1,
                    max(generation_scores),
                )

            generation_best = max(generation_scores)
            history.append(generation_best)
            algorithm.next_generation(generation_scores)
            publish(algorithm.generation, 0, generation_best)
    except Exception as error:
        messages.put(
            {
                "error": f"{type(error).__name__}: {error}",
                "generation": algorithm.generation,
            }
        )
        raise


class EvolutionWorker:
    """Runs fitness trials in a process independent of the display loop."""

    def __init__(self, population_size: int = 16, trial_seconds: float = 5.0) -> None:
        self.population_size = population_size
        self.trial_seconds = trial_seconds
        self._context = multiprocessing.get_context("spawn")
        self._messages = self._context.Queue()
        self._stop_event = self._context.Event()
        self._pause_event = self._context.Event()
        self._process = self._context.Process(
            target=_evolution_worker,
            args=(
                population_size,
                trial_seconds,
                self._messages,
                self._stop_event,
                self._pause_event,
            ),
            name="wheg-evolution",
        )
        self.status: dict = {
            "generation": 1,
            "evaluated": 0,
            "population_size": population_size,
            "best_fitness": None,
            "generation_best": None,
            "history": [],
            "best_genome": None,
            "error": None,
        }

    @property
    def running(self) -> bool:
        return self._process.is_alive()

    def start(self) -> None:
        self._process.start()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._pause_event.set()
        else:
            self._pause_event.clear()

    def poll(self) -> dict:
        while True:
            try:
                self.status.update(self._messages.get_nowait())
            except queue.Empty:
                break

        if self._process.exitcode not in (None, 0) and self.status["error"] is None:
            self.status["error"] = (
                f"Evolution worker exited unexpectedly (code {self._process.exitcode})."
            )
        return self.status

    def stop(self) -> None:
        if self._process.pid is None:
            return
        self._stop_event.set()
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join()
        self._messages.cancel_join_thread()
        self._messages.close()


class Trainer:
    """Runs candidates in bounded batches so the UI remains responsive."""

    TRIAL_SECONDS = 5.0
    STEPS_PER_FRAME = 24

    def __init__(self, population_size: int = 16) -> None:
        self.algorithm = GeneticAlgorithm(population_size=population_size)
        self.candidate_index = 0
        self.scores: list[float] = []
        self.simulation = WhegSimulation(self.algorithm.population[0])
        self.history: list[float] = []
        self.last_generation_best: float | None = None

    @property
    def generation(self) -> int:
        return self.algorithm.generation

    @property
    def evaluating(self) -> int:
        return self.candidate_index + 1

    @property
    def population_size(self) -> int:
        return self.algorithm.population_size

    def update(self, steps: int | None = None) -> None:
        self.simulation.step(self.STEPS_PER_FRAME if steps is None else steps)
        if self.simulation.elapsed + self.simulation.DT / 2.0 < self.TRIAL_SECONDS:
            return

        self.scores.append(self.simulation.fitness())
        self.candidate_index += 1
        if self.candidate_index == self.population_size:
            self.last_generation_best = max(self.scores)
            self.history.append(self.last_generation_best)
            self.algorithm.next_generation(self.scores)
            self.candidate_index = 0
            self.scores.clear()

        self.simulation = WhegSimulation(
            self.algorithm.population[self.candidate_index]
        )

    def reset(self) -> None:
        self.__init__(population_size=self.population_size)
