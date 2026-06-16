import random
import time

import numpy as np


class GreyWolfOptimizer:
    def __init__(self, fitness_func, bounds, population=5, iterations=10):
        self.fitness_func = fitness_func
        self.bounds = bounds
        self.dim = len(bounds)
        self.population = population
        self.iterations = iterations

    def _initialize(self):
        wolves = []
        for _ in range(self.population):
            pos = np.array([random.uniform(bound[0], bound[1]) for bound in self.bounds], dtype=float)
            wolves.append(pos)
        return wolves

    def _evaluate(self, position, iteration, wolf_id):
        try:
            return self.fitness_func(position, iteration=iteration, wolf_id=wolf_id)
        except TypeError:
            return self.fitness_func(position)

    def optimize(self):
        total_start = time.perf_counter()
        wolves = self._initialize()
        fitness = [self._evaluate(w, iteration=1, wolf_id=i + 1) for i, w in enumerate(wolves)]

        pbest_pos = [w.copy() for w in wolves]
        pbest_fit = fitness.copy()

        global_bests = []
        local_bests = []
        iteration_times = []
        evaluation_count = len(wolves)

        for iteration in range(1, self.iterations + 1):
            iter_start = time.perf_counter()

            ranking = np.argsort(fitness)[::-1]
            alpha = wolves[ranking[0]]
            beta = wolves[ranking[1]] if self.population > 1 else wolves[ranking[0]]
            delta = wolves[ranking[2]] if self.population > 2 else wolves[ranking[0]]

            a = 2 - (iteration - 1) * (2 / max(self.iterations - 1, 1))

            for wolf_id in range(self.population):
                position = wolves[wolf_id]

                a1 = 2 * a * np.random.rand(self.dim) - a
                c1 = 2 * np.random.rand(self.dim)
                d_alpha = np.abs(c1 * alpha - position)
                x1 = alpha - a1 * d_alpha

                a2 = 2 * a * np.random.rand(self.dim) - a
                c2 = 2 * np.random.rand(self.dim)
                d_beta = np.abs(c2 * beta - position)
                x2 = beta - a2 * d_beta

                a3 = 2 * a * np.random.rand(self.dim) - a
                c3 = 2 * np.random.rand(self.dim)
                d_delta = np.abs(c3 * delta - position)
                x3 = delta - a3 * d_delta

                new_position = (x1 + x2 + x3) / 3.0
                for dimension in range(self.dim):
                    lower, upper = self.bounds[dimension]
                    new_position[dimension] = np.clip(new_position[dimension], lower, upper)
                wolves[wolf_id] = new_position

            fitness = []
            for wolf_id, wolf in enumerate(wolves, start=1):
                score = self._evaluate(wolf, iteration=iteration, wolf_id=wolf_id)
                fitness.append(score)
                evaluation_count += 1

            for wolf_index in range(self.population):
                if fitness[wolf_index] > pbest_fit[wolf_index]:
                    pbest_fit[wolf_index] = fitness[wolf_index]
                    pbest_pos[wolf_index] = wolves[wolf_index].copy()

            local_bests.append(pbest_fit.copy())
            global_bests.append(float(np.max(pbest_fit)))
            iteration_times.append(time.perf_counter() - iter_start)

        best_index = int(np.argmax(pbest_fit))
        best_pos = pbest_pos[best_index]
        total_time = time.perf_counter() - total_start

        return {
            "best_pos": best_pos,
            "best_conf": self._map_position(best_pos),
            "best_fitness": float(pbest_fit[best_index]),
            "best_index": best_index + 1,
            "global_bests": global_bests,
            "local_bests": local_bests,
            "iteration_times": iteration_times,
            "total_optimization_time": total_time,
            "evaluation_count": evaluation_count,
        }

    def _map_position(self, pos):
        return pos
