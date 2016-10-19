import time

from tsp import TSP
import numpy as np


class TSPSimulatedAnnealing(TSP):

    def __init__(self, distance_matrix, initial_solution_strategy='greedy', neighbor_selection='first',
                 neighborhood='2-opt', t_max=100, t_min=0.01, cooling_function=(lambda (t, i): t * 0.99),
                 max_iterations_at_each_t=50):
        TSP.__init__(self, distance_matrix, initial_solution_strategy, neighbor_selection, neighborhood)
        self.t_max = float(t_max)
        self.t_min = float(t_min)
        self.cooling_function = cooling_function
        self.max_iterations_at_each_t = max_iterations_at_each_t

    def run(self, max_iter, verbose=None):
        """
        :param max_iter: int
            Number of iterations of the local search.
        :return: self.current_solution_
            Final solution of the algorithm.
        :return: self.current_cost_
            Costs of the final solution.
        """
        number_of_iterations = 0
        time_init = time.time()
        t = self.t_max
        while number_of_iterations < max_iter and t > self.t_min:
            iterations = 0
            while iterations < self.max_iterations_at_each_t:
                i = np.random.randint(0, self.number_of_cities - 1)
                j = np.random.randint(i + 1, self.number_of_cities)
                random_neighbor = self._neighbor_2_opt(self.current_solution, i, j)
                random_neighbor_cost = self._evaluate_solution(random_neighbor)
                delta_e = random_neighbor_cost - self.current_cost
                if delta_e < 0:
                    self.current_solution = random_neighbor
                    self.current_cost = random_neighbor_cost
                else:
                    p = np.exp(-delta_e / t)
                    accept = np.random.choice([True, False], p=[p, 1-p])
                    if accept:
                        self.current_solution = random_neighbor
                        self.current_cost = random_neighbor_cost
                iterations += 1
            if verbose and number_of_iterations % verbose == 0 and number_of_iterations != 0:
                time_elapsed = time.time() - time_init
                self._report(number_of_iterations, time_elapsed, t)
            number_of_iterations += 1
            t = self.cooling_function((t, iterations))


        time_elapsed = time.time() - time_init
        self._report(number_of_iterations, time_elapsed, t)
        return self.current_solution, self.current_cost

    def _report(self, n_iteration, time_elapsed, t):
        print "Iteration number", n_iteration
        print "\t * Current solution:", list(self.current_solution)
        print "\t * Current cost:", self.current_cost
        print "\t * Current temperature:", t
        print "\t * Time elapsed: %.2f seconds" % time_elapsed


