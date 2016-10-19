import time
from tsp import TSP


class TSPLocalSearch(TSP):

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
        while number_of_iterations < max_iter:
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

        time_elapsed = time.time() - time_init
        self._report(number_of_iterations, time_elapsed)

        return self.current_solution, self.current_cost

    def _report(self, n_iteration, time_elapsed):
        print "Iteration number", n_iteration
        print "\t * Current solution:", list(self.current_solution)
        print "\t * Current cost:", self.current_cost
        print "\t * Time elpased:", time_elapsed, 'seconds'
