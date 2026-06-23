import time

import numpy as np

from .base_optimizer import BaseOptimizer


class ParticleSwarmOptimizer(BaseOptimizer):
    def optimize(self):
        total_start = time.perf_counter()
        positions = self._initialize()
        velocities = np.zeros_like(positions)
        fitness = [self._evaluate(position, iteration=0, agent_id=index + 1) for index, position in enumerate(positions)]
        pbest_pos = positions.copy()
        pbest_fit = np.array(fitness, dtype=float)
        gbest_index = int(np.argmax(pbest_fit))
        gbest_pos = pbest_pos[gbest_index].copy()
        global_bests = []
        local_bests = []
        iteration_times = []
        iteration_summaries = [self._summarize_iteration(0, fitness, positions, pbest_fit, total_start)]
        evaluation_count = len(positions)

        w_max, w_min = 0.9, 0.4
        c1, c2 = 1.7, 1.7

        for iteration in range(1, self.iterations + 1):
            iter_start = time.perf_counter()
            inertia = w_max - (w_max - w_min) * ((iteration - 1) / max(self.iterations - 1, 1))

            for particle_index in range(self.population):
                r1 = self.random_state.random(self.dim)
                r2 = self.random_state.random(self.dim)
                velocities[particle_index] = inertia * velocities[particle_index] + c1 * r1 * (pbest_pos[particle_index] - positions[particle_index]) + c2 * r2 * (gbest_pos - positions[particle_index])
                positions[particle_index] = self._clip(positions[particle_index] + velocities[particle_index])

            fitness = [self._evaluate(position, iteration=iteration, agent_id=index + 1) for index, position in enumerate(positions)]
            evaluation_count += len(positions)

            for particle_index in range(self.population):
                if fitness[particle_index] > pbest_fit[particle_index]:
                    pbest_fit[particle_index] = fitness[particle_index]
                    pbest_pos[particle_index] = positions[particle_index].copy()

            gbest_index = int(np.argmax(pbest_fit))
            gbest_pos = pbest_pos[gbest_index].copy()
            local_bests.append(pbest_fit.copy())
            global_bests.append(float(np.max(pbest_fit)))
            iteration_times.append(time.perf_counter() - iter_start)
            iteration_summaries.append(self._summarize_iteration(iteration, fitness, positions, pbest_fit, iter_start))

        return {
            "best_pos": gbest_pos,
            "best_conf": gbest_pos,
            "best_fitness": float(np.max(pbest_fit)),
            "best_index": gbest_index + 1,
            "global_bests": global_bests,
            "local_bests": local_bests,
            "iteration_times": iteration_times,
            "iteration_summaries": iteration_summaries,
            "total_optimization_time": time.perf_counter() - total_start,
            "evaluation_count": evaluation_count,
        }
