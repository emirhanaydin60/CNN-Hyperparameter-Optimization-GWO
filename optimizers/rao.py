import time

import numpy as np

from .base_optimizer import BaseOptimizer


class RaoOptimizer(BaseOptimizer):
    def optimize(self):
        total_start = time.perf_counter()
        agents = self._initialize()
        fitness = [self._evaluate(agent, iteration=0, agent_id=index + 1) for index, agent in enumerate(agents)]
        pbest_pos = agents.copy()
        pbest_fit = np.array(fitness, dtype=float)
        global_bests = []
        local_bests = []
        iteration_times = []
        iteration_summaries = [self._summarize_iteration(0, fitness, agents, pbest_fit, total_start)]
        evaluation_count = len(agents)

        for iteration in range(1, self.iterations + 1):
            iter_start = time.perf_counter()
            best_index = int(np.argmax(pbest_fit))
            worst_index = int(np.argmin(pbest_fit))
            best_pos = pbest_pos[best_index].copy()
            worst_pos = pbest_pos[worst_index].copy()

            for agent_index in range(self.population):
                rand1 = self.random_state.random(self.dim)
                rand2 = self.random_state.random(self.dim)
                rand3 = self.random_state.random(self.dim)
                if self.random_state.random() < 0.5:
                    candidate = agents[agent_index] + rand1 * (best_pos - np.abs(agents[agent_index])) - rand2 * (worst_pos - np.abs(agents[agent_index]))
                else:
                    peer_index = int(self.random_state.integers(0, self.population))
                    candidate = agents[agent_index] + rand3 * (agents[peer_index] - agents[agent_index])
                agents[agent_index] = self._clip(candidate)

            fitness = [self._evaluate(agent, iteration=iteration, agent_id=index + 1) for index, agent in enumerate(agents)]
            evaluation_count += len(agents)

            for agent_index in range(self.population):
                if fitness[agent_index] > pbest_fit[agent_index]:
                    pbest_fit[agent_index] = fitness[agent_index]
                    pbest_pos[agent_index] = agents[agent_index].copy()

            local_bests.append(pbest_fit.copy())
            global_bests.append(float(np.max(pbest_fit)))
            iteration_times.append(time.perf_counter() - iter_start)
            iteration_summaries.append(self._summarize_iteration(iteration, fitness, agents, pbest_fit, iter_start))

        best_index = int(np.argmax(pbest_fit))
        best_pos = pbest_pos[best_index].copy()
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
