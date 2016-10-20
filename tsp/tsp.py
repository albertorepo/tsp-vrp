from copy import deepcopy

from basic import Problem
import random
import numpy as np


class TSP(Problem):
    def __init__(self, distance_matrix, initial_solution_strategy='greedy', neighbor_selection='first',
                 neighborhood='2-opt', seed=None):
        self.distance_matrix = distance_matrix
        self.number_of_cities = len(self.distance_matrix)
        self.seed = seed
        self.current_solution = self._get_initial_solution(initial_solution_strategy)
        self.current_cost = self._evaluate_solution(self.current_solution)
        self.neighbor_selection = neighbor_selection
        self.neighborhood = neighborhood

    def _get_initial_solution(self, strategy):
        if strategy == 'random':
            solution = range(1, self.number_of_cities + 1)
            random.seed(self.seed)
            random.shuffle(solution)
            return solution
        elif strategy == 'greedy':
            bag_of_cities = range(0, self.number_of_cities)
            random.seed(self.seed)
            solution = [random.randint(1, self.number_of_cities)]
            bag_of_cities.remove(solution[0] - 1)
            for i in range(1, self.number_of_cities):
                distances = [self.distance_matrix[solution[i - 1] - 1][j] for j in bag_of_cities]
                solution.append(bag_of_cities[np.argmin(distances)] + 1)
                bag_of_cities.remove(solution[i] - 1)
            return solution

        return None

    def _neighbor_2_opt(self, solution, i, j):
        solution_tmp = deepcopy(solution)
        solution_tmp[i - 1:j] = list(reversed(solution[i - 1:j]))
        return solution_tmp

    def _neighbor_swap(self, solution, i, j):
        solution_tmp = deepcopy(solution)
        solution_tmp[i], solution_tmp[j] = solution_tmp[j], solution_tmp[i]
        return solution_tmp

    def _evaluate_solution(self, solution, idx=1):
        if idx == len(solution):
            return self._distance(solution[idx - 1], solution[0])
        else:
            return self._distance(solution[idx], solution[idx - 1]) + self._evaluate_solution(solution, idx=idx + 1)

    def _distance(self, a, b):
        return self.distance_matrix[a - 1][b - 1]

    def _select_neighbor(self, *args, **kwargs):
        raise NotImplementedError

    def _report(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError
