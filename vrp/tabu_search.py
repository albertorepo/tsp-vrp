import time

from cvrp import CVRP


class CVRPTabuSearch(CVRP):

    def __init__(self, distance_matrix, initial_solution_strategy='greedy', neighbor_selection='first',
                 neighborhood='2-opt', seed=None, tabu_size=None):
        CVRP.__init__(self, distance_matrix, initial_solution_strategy, neighbor_selection, neighborhood, seed)
        self.tabu_size = tabu_size
        self.tabu = [self.current_solution]
        self.best_solution = self.current_solution


    def run(self, max_iter, verbose=None):
        number_of_iterations = 0
        time_init = time.time()
        if not self.tabu_size:
            self.tabu_size = max_iter / 10

        while number_of_iterations < max_iter:
            self.prune_tabu_list()
            best_neighbor, best_neighbor_cost = self._select_neighbor(self.current_solution)
            self.current_solution = best_neighbor
            self.current_cost = best_neighbor_cost

            if best_neighbor_cost < self.best_cost:
                self.update_best_solution(best_neighbor, best_neighbor_cost)

            if verbose and number_of_iterations % verbose == 0 and number_of_iterations != 0:
                time_elapsed = time.time() - time_init
                self._report(number_of_iterations, time_elapsed)

            number_of_iterations += 1
            self.tabu.append(self.current_solution)

        time_elapsed = time.time() - time_init
        self._report(number_of_iterations, time_elapsed)
        return self.best_solution, self.best_cost


    def _select_neighbor(self, solution):

        neighbors = self._neighborhood(solution)
        tweaked_neighbors = [neighbor for neighbor in neighbors if neighbor not in self.tabu]
        costs = [self._evaluate_solution(neighbor) for neighbor in tweaked_neighbors]
        best_neighbor, best_neighbor_cost = neighbors[np.argmin(costs)], np.min(costs)
        return best_neighbor, best_neighbor_cost

    def _neighborhood(self, solution):
        neighbors = []
        idx = 0
        for
    def prune_tabu_list(self):
        while len(self.tabu) > self.tabu_size:
            self.tabu.pop(0)

    def update_best_solution(self, solution, cost):
        self.best_solution = solution
        self.best_cost = cost

    def _report(self, n_iteration, time_elapsed):
        print "Iteration number", n_iteration
        print "\t * Current solution:", list(self.current_solution)
        print "\t * Current cost:", self.current_cost
        print "\t * Time elpased:", time_elapsed, 'seconds'
        print "\t * Tabu list length:", len(self.tabu)