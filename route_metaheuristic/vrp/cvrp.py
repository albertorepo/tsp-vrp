import itertools
import math
import random
from copy import deepcopy

import numpy as np

from route_metaheuristic.basic import Problem



class CVRP(Problem):
    def __init__(self, distance_matrix, demand, truck_capacity, depot=0, initial_solution=None,
                 initial_solution_strategy='random', neighborhood='2-opt-star'):
        self.distance_matrix = distance_matrix
        self.number_of_cities = len(self.distance_matrix)
        self.demand = demand
        self.truck_capacity = truck_capacity
        self.depot = depot
        self.initial_solution_strategy = initial_solution_strategy
        if initial_solution:
            self.current_solution = initial_solution
        else:
            self.current_solution = self._get_initial_solution()
        self.neighborhood = neighborhood

    def _get_initial_solution(self):
        if self.initial_solution_strategy == 'sequential':
            return self._seq_initial_solution()
        elif self.initial_solution_strategy == 'random':
            return self._rand_initial_solution()
        else:
            raise AttributeError('`initial_solution_strategy` must be either `sequential` or `random`')

    def _rand_initial_solution(self):
        cities = range(1, self.number_of_cities)
        max_cities = int(math.ceil(self.truck_capacity / float(max(self.demand))))

        random.shuffle(cities)
        solution = []
        while cities:
            random_n_cities = random.randint(1, max_cities)
            solution.append([0] + cities[0:random_n_cities])
            cities = cities[random_n_cities:]

        return solution

    def _seq_initial_solution(self):
        """Sequential route building"""
        cities = range(1, self.number_of_cities)
        solution = []
        while cities:
            solution.append([0])
            while cities:
                demand = sum([self.demand[i] for i in solution[-1]])
                distances_of_last_city = self.distance_matrix[solution[-1][-1]]
                distances_of_feasible_cities = [distances_of_last_city[i] for i in cities]
                nearest_city = cities[np.argmin(distances_of_feasible_cities)]
                if demand + self.demand[nearest_city] > self.truck_capacity:
                    break
                solution[-1].append(nearest_city)
                cities.remove(nearest_city)

        return solution

    def _select_neighbor(self, solution):
        if self.neighborhood == '2-opt-star':
            return self._neighborhood_2_opt_star(solution)
        elif self.neighborhood == '2-opt':
            return self._neighborhood_2_opt(solution)

    def _neighborhood_2_opt_star(self, solution):
        solution_tmp = deepcopy(solution)
        best_incremental_cost = 0
        for (r_n, R), (t_n, T) in itertools.combinations(enumerate(solution_tmp), 2):
            for u in range(0, len(R)):
                for v in range(0, len(T)):
                    if self._2opt_change_is_feasible(R, T, u, v):
                        incremental_cost = self._increment_of_cost_between_routes(R, T, u, v)
                        if incremental_cost < best_incremental_cost:
                            best_incremental_cost = incremental_cost
                            best_u = u
                            best_v = v
            if best_incremental_cost < 0:
                return self._neighbor_2_opt_star(solution_tmp, R, T, r_n, t_n, best_u, best_v)
        return None

    def _neighbor_2_opt_star(self, solution, route1, route2, n1, n2, u, v):
        route1_tmp = route1[0:u + 1] + route2[v + 1:]
        route2_tmp = route2[0:v + 1] + route1[u + 1:]
        solution[n1] = route1_tmp
        solution[n2] = route2_tmp
        return [route for route in solution if len(route) > 1]

    def _neighborhood_2_opt(self, solution):
        solution_tmp = deepcopy(solution)
        for n, route in enumerate(solution_tmp):
            for i in range(1, len(route)):
                for j in range(i + 1, len(route)):
                    neighbor = self._neighbor_2_opt(route, i, j)
                    if self._route_cost(neighbor) < self._route_cost(route):
                        solution_tmp[n] = neighbor
                        return solution_tmp
        return None

    def _neighbor_2_opt(self, route, i, j):
        solution_tmp = deepcopy(route)
        solution_tmp[i - 1:j] = list(reversed(route[i - 1:j]))
        return solution_tmp

    def _load(self, route, i=0, j=None):
        if not j:
            j = len(route) - 1

        return sum([self.demand[route[k]] for k in range(i, j + 1)])

    def _route_cost(self, route, i=0, j=None):
        if not j:
            j = len(route)
        dist = 0
        for k in range(i, j):
            if k == len(route) - 1:
                dist += self.distance_matrix[route[k]][route[0]]
            else:
                dist += self.distance_matrix[route[k]][route[k + 1]]
        return dist

    def _inter_route_cost(self, route1, route2, node1, node2):
        if node1 == len(route1):
            node1 = 0
        if node2 == len(route2):
            node2 = 0
        return self.distance_matrix[route1[node1]][route2[node2]]

    def _increment_of_cost_between_routes(self, route1, route2, u, v):
        return self._inter_route_cost(route1, route2, u, v + 1) + \
               self._inter_route_cost(route2, route1, v, u + 1) - \
               self._route_cost(route1, u, u + 1) - \
               self._route_cost(route2, v, v + 1)

    def _2opt_change_is_feasible(self, route1, route2, u, v):
        load_1_until_u = sum(self.demand[route1[k]] for k in range(u + 1))
        load_2_until_v = sum(self.demand[route2[k]] for k in range(v + 1))
        return load_1_until_u + self._load(route2) - load_2_until_v <= self.truck_capacity and \
               load_2_until_v + self._load(route1) - load_1_until_u <= self.truck_capacity

    def _evaluate_solution(self, solution):
        return sum([self._route_cost(route) for route in solution])

    def run(self, *args, **kwargs):
        raise NotImplementedError
