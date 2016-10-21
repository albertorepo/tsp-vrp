import time
import math
from cvrp import CVRP
import random
import numpy as np


class CVRPSimulatedAnnealing(CVRP):

    def __init__(self, distance_matrix, demand, truck_capacity, depot=0, initial_solution=None,
                 initial_solution_strategy='random', neighborhood='2-opt-star', t_max=100, t_min=0.01,
                 cooling_function=(lambda (t, i): t * 0.99), max_iterations_at_each_t=50):
        CVRP.__init__(self, distance_matrix, demand, truck_capacity, depot, initial_solution, initial_solution_strategy,
                      neighborhood)
        self.t_max = float(t_max)
        self.t_min = float(t_min)
        self.cooling_function = cooling_function
        self.max_iterations_at_each_t = max_iterations_at_each_t

    def run(self, max_iter, verbose=None):
        number_of_iterations = 0
        time_init = time.time()
        t = self.t_max
        while number_of_iterations < max_iter and t > self.t_min:
            iterations = 0
            while iterations < self.max_iterations_at_each_t:
                random_neighbor, delta_e = self._get_random_feasible_neighbor()

                if delta_e < 0:
                    self.current_solution = random_neighbor
                else:
                    p = math.exp(-delta_e / t)
                    accept = np.random.choice([True, False], p=[p, 1 - p])
                    if accept:
                        self.current_solution = random_neighbor
                iterations += 1
            if verbose and number_of_iterations % verbose == 0 and number_of_iterations != 0:
                time_elapsed = time.time() - time_init
                self._report(number_of_iterations, time_elapsed, t)
            number_of_iterations += 1
            t = self.cooling_function((t, iterations))

        time_elapsed = time.time() - time_init
        self._report(number_of_iterations, time_elapsed, t)
        return self.current_solution, self._evaluate_solution(self.current_solution)

    def _get_random_feasible_neighbor(self):
        while True:
            route1, route2 = random.sample(self.current_solution, 2)
            u = random.randint(0, len(route1) - 1)
            v = random.randint(0, len(route2) - 1)
            if self._2opt_change_is_feasible(route1, route2, u, v):
                break
        r1_n, r2_n = self.current_solution.index(route1), self.current_solution.index(route2)
        delta_e = self._increment_of_cost_between_routes(route1, route2, u, v)
        return self._neighbor_2_opt_star(self.current_solution, route1, route2, r1_n, r2_n, u, v), delta_e

    def _report(self, n_iteration, time_elapsed, t):
        print "Iteration number", n_iteration
        print "\t * Current solution:", self.current_solution
        print "\t * Current cost:", self._evaluate_solution(self.current_solution)
        print "\t * Current temperature:", t
        print "\t * Time elapsed: %.2f seconds" % time_elapsed

