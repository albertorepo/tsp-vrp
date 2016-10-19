from copy import deepcopy

from basic import Problem
from utils import unique
from random import shuffle
import numpy as np


class TSP(Problem):

    def __init__(self, distance_matrix, initial_solution_strategy='greedy', neighbor_selection='first',
                 neighborhood='2-opt'):
        self.distance_matrix = distance_matrix
        self.number_of_cities = len(self.distance_matrix)
        self.current_solution = self._get_initial_solution(initial_solution_strategy)
        self.current_cost = self._evaluate_solution(self.current_solution)
        self.neighbor_selection = neighbor_selection
        self.neighborhood = neighborhood

    def _get_initial_solution(self, strategy):
        if strategy == 'random':
            solution = range(1, self.number_of_cities + 1)
            shuffle(solution)
            return solution
        elif strategy == 'greedy':
            bag_of_cities = range(0,self.number_of_cities)
            solution = [np.random.randint(1, self.number_of_cities)]
            bag_of_cities.remove(solution[0] - 1)
            for i in range(1,self.number_of_cities):
                distances = [self.distance_matrix[solution[i-1] - 1][j] for j in bag_of_cities]
                solution.append(bag_of_cities[np.argmin(distances)] + 1)
                bag_of_cities.remove(solution[i] - 1)
            return solution

        return None

    def _neighborhood(self, solution):
        if self.neighborhood == '2-opt':
            return self._neighborhood_2_opt(solution)
        elif self.neighborhood == 'swap':
            return self._neighbor_swap(solution)

    def _neighborhood_2_opt(self, solution):
        neighbors = [[None for i in xrange(self.number_of_cities)] for j in xrange(self.number_of_cities * (self.number_of_cities -1) / 2)]
        idx = 0
        for i in range(self.number_of_cities):
            for j in range(i + 1, self.number_of_cities):
                neighbors[idx] = self._neighbor_2_opt(solution, i, j)
                idx += 1
        return unique(neighbors)

    def _neighbor_2_opt(self, solution, i, j):
        solution_tmp = deepcopy(solution)
        solution_tmp[i - 1:j] = list(reversed(solution[i - 1:j]))
        return solution_tmp

    def _neighbor_swap(self, solution):
        neighbors = [[None for i in xrange(self.number_of_cities)] for j in xrange(self.number_of_cities - 1 / 2)]
        idx = 0
        for i in range(self.number_of_cities):
            for j in range(i + 1, self.number_of_cities):
                solution_tmp = deepcopy(solution)
                solution_tmp[i], solution_tmp[j] = solution_tmp[j], solution_tmp[i]
                neighbors[idx] = solution_tmp
        return neighbors

    def _evaluate_solution(self, solution, idx=1):
        if idx == len(solution):
            return self._distance(solution[idx - 1], solution[0])
        else:
            return self._distance(solution[idx], solution[idx - 1]) + self._evaluate_solution(solution, idx=idx + 1)

    def _distance(self, a, b):
        return self.distance_matrix[a - 1][b - 1]

    def _select_neighbor(self, neighbors):
        if self.neighbor_selection == 'best':
            return self._best_neighbor(neighbors)
        elif self.neighbor_selection == 'first':
            return self._first_neighbor(neighbors)
        raise AttributeError('`neighbor_selection` must be either `best` or `first`')

    def _best_neighbor(self, neighbors):
        costs = [self._evaluate_solution(neighbor) for neighbor in neighbors]
        best_neighbor, best_cost = neighbors[np.argmin(costs)], np.min(costs)
        if best_cost >= self.current_cost:
            return None, None
        return best_neighbor, best_cost

    def _first_neighbor(self, neighbors):
        for neighbor in  neighbors:
            cost = self._evaluate_solution(neighbor)
            if cost < self.current_cost:
                return neighbor, cost
        return None, None

    def _report(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError