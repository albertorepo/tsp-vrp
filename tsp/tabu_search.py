import time
import numpy as np

from tsp import TSP


class TSPTabuSearch(TSP):

    def __init__(self, distance_matrix, initial_solution_strategy='greedy', neighbor_selection='first',
                 neighborhood='2-opt', tabu_size=None, max_threshold=100):
        TSP.__init__(self, distance_matrix, initial_solution_strategy, neighbor_selection, neighborhood)
        self.tabu_size = tabu_size
        self.tabu = [self.current_solution]
        self.max_threshold = max_threshold

    def run(self, max_iter, verbose=None):
        if not self.tabu_size:
            self.tabu_size = max_iter / 10
        number_of_iterations = 0
        time_init = time.time()
        while number_of_iterations < max_iter:
            self.prune_tabu_list()
            neighbors = self._neighborhood_2_opt(self.current_solution)
            best_neighbor, best_cost = self._select_neighbor(neighbors)
            if best_neighbor is None:
                break
            self.current_solution = best_neighbor
            self.current_cost = best_cost
            if verbose and number_of_iterations % verbose == 0 and number_of_iterations != 0:
                time_elapsed = time.time() - time_init
                self._report(number_of_iterations, time_elapsed)
            number_of_iterations += 1
            self.tabu.append(self.current_solution)

        time_elapsed = time.time() - time_init
        self._report(number_of_iterations, time_elapsed)

        return self.current_solution, self.current_cost

    def _select_neighbor(self, neighbors):
        tweaked_neighbors = [neighbor for neighbor in neighbors if neighbor not in self.tabu]
        costs = [self._evaluate_solution(neighbor) for neighbor in tweaked_neighbors]
        best_neighbor, best_cost = neighbors[np.argmin(costs)], np.min(costs)
        if best_cost - self.current_cost > self.max_threshold:
            return None, None
        return best_neighbor, best_cost

    def _report(self, n_iteration, time_elapsed):
        print "Iteration number", n_iteration
        print "\t * Current solution:", list(self.current_solution)
        print "\t * Current cost:", self.current_cost
        print "\t * Time elpased:", time_elapsed, 'seconds'

    def prune_tabu_list(self):
        while len(self.tabu) > self.tabu_size:
            self.tabu.pop(0)