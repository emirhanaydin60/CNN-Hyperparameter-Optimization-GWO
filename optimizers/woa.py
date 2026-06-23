import time

import numpy as np

from .base_optimizer import BaseOptimizer


class WhaleOptimizationOptimizer(BaseOptimizer):
    def optimize(self):
        total_start = time.perf_counter()
        whales = self._initialize()
        fitness = [self._evaluate(whale, iteration=0, agent_id=index + 1) for index, whale in enumerate(whales)]
        pbest_pos = whales.copy()
        pbest_fit = np.array(fitness, dtype=float)
        best_index = int(np.argmax(pbest_fit))
        best_pos = pbest_pos[best_index].copy()
        global_bests = []
        local_bests = []
        iteration_times = []
        iteration_summaries = [self._summarize_iteration(0, fitness, whales, pbest_fit, total_start)]
        evaluation_count = len(whales)

        for iteration in range(1, self.iterations + 1):
            iter_start = time.perf_counter()
            a = 2 - 2 * ((iteration - 1) / max(self.iterations - 1, 1))

            for whale_index in range(self.population):
                current = whales[whale_index]
                r1 = self.random_state.random(self.dim)
                r2 = self.random_state.random(self.dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                p = self.random_state.random()

                if p < 0.5:
                    if np.abs(A).max() < 1:
                        distance = np.abs(C * best_pos - current)
                        candidate = best_pos - A * distance
                    else:
                        random_whale = whales[self.random_state.integers(0, self.population)]
                        distance = np.abs(C * random_whale - current)
                        candidate = random_whale - A * distance
                else:
                    distance = np.abs(best_pos - current)
                    l = self.random_state.uniform(-1, 1)
                    candidate = distance * np.exp(l) * np.cos(2 * np.pi * l) + best_pos

                whales[whale_index] = self._clip(candidate)

            fitness = [self._evaluate(whale, iteration=iteration, agent_id=index + 1) for index, whale in enumerate(whales)]
            evaluation_count += len(whales)

            for whale_index in range(self.population):
                if fitness[whale_index] > pbest_fit[whale_index]:
                    pbest_fit[whale_index] = fitness[whale_index]
                    pbest_pos[whale_index] = whales[whale_index].copy()

            best_index = int(np.argmax(pbest_fit))
            best_pos = pbest_pos[best_index].copy()
            local_bests.append(pbest_fit.copy())
            global_bests.append(float(np.max(pbest_fit)))
            iteration_times.append(time.perf_counter() - iter_start)
            iteration_summaries.append(self._summarize_iteration(iteration, fitness, whales, pbest_fit, iter_start))

        return {
            "best_pos": best_pos,
            "best_conf": best_pos,
            "best_fitness": float(np.max(pbest_fit)),
            "best_index": best_index + 1,
            "global_bests": global_bests,
            "local_bests": local_bests,
            "iteration_times": iteration_times,
            "iteration_summaries": iteration_summaries,
            "total_optimization_time": time.perf_counter() - total_start,
            "evaluation_count": evaluation_count,
        }
