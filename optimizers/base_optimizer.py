import time

import numpy as np


class BaseOptimizer:
    def __init__(self, fitness_func, bounds, population=8, iterations=15, solution_signature_func=None, random_seed=None):
        self.fitness_func = fitness_func
        self.bounds = bounds
        self.dim = len(bounds)
        self.population = population
        self.iterations = iterations
        self.solution_signature_func = solution_signature_func
        self.random_state = np.random.default_rng(random_seed)

    def _initialize(self):
        return np.array(
            [[self.random_state.uniform(low, high) for low, high in self.bounds] for _ in range(self.population)],
            dtype=float,
        )

    def _evaluate(self, position, iteration, agent_id):
        try:
            return self.fitness_func(position, iteration=iteration, wolf_id=agent_id)
        except TypeError:
            try:
                return self.fitness_func(position, iteration=iteration, agent_id=agent_id)
            except TypeError:
                return self.fitness_func(position)

    def _signature(self, position):
        if self.solution_signature_func is not None:
            return self.solution_signature_func(position)
        return tuple(np.round(position, 6).tolist())

    def _clip(self, position):
        clipped = position.copy()
        for dimension, (lower, upper) in enumerate(self.bounds):
            clipped[dimension] = np.clip(clipped[dimension], lower, upper)
        return clipped

    def _summarize_iteration(self, iteration, fitness, positions, pbest_fit, iteration_start):
        current_signatures = [self._signature(position) for position in positions]
        diversity = float(np.mean(np.std(np.asarray(positions, dtype=float), axis=0))) if len(positions) > 1 else 0.0
        best_fitness = float(np.max(fitness)) if len(fitness) else 0.0
        worst_fitness = float(np.min(fitness)) if len(fitness) else 0.0
        global_best = float(np.max(pbest_fit)) if len(pbest_fit) else 0.0
        exploration_ratio = float(np.mean(np.asarray(fitness, dtype=float) < global_best)) if len(fitness) else 0.0
        return {
            "iteration": iteration,
            "unique_solution_count": len(set(current_signatures)),
            "repeat_rate": 1.0 - (len(set(current_signatures)) / max(len(current_signatures), 1)),
            "average_fitness": float(np.mean(fitness)) if len(fitness) else 0.0,
            "best_fitness": best_fitness,
            "worst_fitness": worst_fitness,
            "population_diversity": diversity,
            "exploration_ratio": exploration_ratio,
            "exploitation_ratio": 1.0 - exploration_ratio,
            "iteration_time": time.perf_counter() - iteration_start,
        }
