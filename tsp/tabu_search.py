import time
import numpy as np
from utils import unique

from tsp import TSP


class TSPTabuSearch(TSP):
    """Traveling Salesman Problem.

    Parameters
    ----------
    distance_matrix : array-like, shape = (n_cities, n_cities)
        Relative distances or costs between cities of the problem.
        Main diagonal elements of the matrix are expected to be 0.

    initial_solution_strategy : string, optional, (default='greedy')
        Strategy for creating an initial solution. Supported strategies
        are 'greedy' for a greedy constructional solution and 'random' for
        a random generated solution.

    neighbor_selection : string, optional, (default='first')
        Strategy for selecting a neighbor among the solution neighborhood.
        Supported strategies are 'first' for the first improving neighbor and
        'best' for the best improving neighbor.

        - If greedy, a initial solution that minimize the distance between
        consecutive cities will be provided.
        - If random, a random permutation of the cities of the problem will
        be built.

    neighborhood : string, optional, (default='2-opt')
        Neighborhood to explore at each iteration of the algorithm. Supported
        neighborhoods are '2-opt' for the 2-opt neighborhood and 'swap' for
        the swap movements neighborhood.

    seed : int, optional, (default=None)
        Random seed for every randomized choice.

    tabu_size : int
        Size of the tabu list. Maximum number of solutions to forbid.



    Attributes
    ----------
    number_of_cities_ : int
        Number of cities in the problem instance.

    current_solution_: array-like, shape = (n_cities)
        At the moment of the query, the current solution of the problem.
        This attribute can be queried after the creation of the object or
        after the algorithm (run method) has been called.

    current_cost_ : float
        Objective function value for current_solution_ attribute

    tabu_ : array-like, shape = (tabu_size, n_cities)
        Actual tabu list. Set of solutions that are forbidden.

    best_solution_ : array-like, shape = (n_cities)
        Best solution found so far.

    best_cost_ : float
        Objective function value of the best solution.



    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Travelling_salesman_problem

    .. [2] https://en.wikipedia.org/wiki/2-opt

    .. [3] Gendreau, M., Hertz, A., & Laporte, G. (1994). A tabu search
    heuristic for the vehicle routing problem. Management science,
    40(10), 1276-1290.

    """
    def __init__(self,
                 distance_matrix,
                 initial_solution_strategy='greedy',
                 neighbor_selection='first',
                 neighborhood='2-opt',
                 seed=None,
                 tabu_size=None):
        TSP.__init__(
            self,
            distance_matrix,
            initial_solution_strategy,
            neighbor_selection,
            neighborhood,
            seed)
        self.tabu_size = tabu_size
        self.tabu_ = [self.current_solution]
        self.best_solution_ = self.current_solution
        self.best_cost_ = self.current_cost

    def run(self, max_iter, verbose=None):
        """Run a simulated annealing search over the algorithm instance.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations of the algorithm. Local
            search will be stopped if not better solution is found
            at current iterations or if the maximum iterations
            number is reached.

        verbose: int or None, optional, (defaul=None)
            If `verbose` is an integer then a summary of the solution
            is printed at every given `verbose` iteration.
        """
        number_of_iterations = 0
        time_init = time.time()
        if not self.tabu_size:
            self.tabu_size = max_iter / 10

        while number_of_iterations < max_iter:
            self.prune_tabu_list()
            best_neighbor, best_neighbor_cost = self._select_neighbor(self.current_solution)
            self.current_solution = best_neighbor
            self.current_cost = best_neighbor_cost

            if best_neighbor_cost < self.best_cost_:
                self.update_best_solution(best_neighbor, best_neighbor_cost)

            if verbose and number_of_iterations % verbose == 0 and number_of_iterations != 0:
                time_elapsed = time.time() - time_init
                self._report(number_of_iterations, time_elapsed)

            number_of_iterations += 1
            self.tabu_.append(self.current_solution)

        time_elapsed = time.time() - time_init
        self._report(number_of_iterations, time_elapsed)
        return self.best_solution_, self.best_cost_

    def _select_neighbor(self, solution):
        """Select neighbor from solution to set as the new solution.

        The neighborhood to explore is set in the neighborhood
        parameter.

        Tabu list will be checked in order to ensure that the
        neighbor is not in it.

        Parameters
        ----------
        solution : array-like, shape = (n_cities)
            Original solution whose neighborhood will be explored.

        Returns
        -------
        best_neighbor : array-like, shape = (n_cities)
            Returns neighbor to return.
        best_cost : float
            Returns cost of the best_neighbor
        """

        if self.neighborhood == '2-opt':
            get_neighbor = self._neighbor_2_opt
        elif self.neighborhood == 'swap':
            get_neighbor = self._neighbor_swap
        else:
            raise AttributeError

        neighbors = self._neighborhood(solution, get_neighbor)
        tweaked_neighbors = [neighbor for neighbor in neighbors if neighbor not in self.tabu_]
        costs = [self._evaluate_solution(neighbor) for neighbor in tweaked_neighbors]
        best_neighbor, best_neighbor_cost = neighbors[np.argmin(costs)], np.min(costs)
        return best_neighbor, best_neighbor_cost

    def _neighborhood(self, solution, get_neighbor):
        """Get the neighborhood of a solution.

        Parameters
        ----------
        solution : array-like, shape = (n_cities)
            Original solution whose neighborhood will be explored.

        get_neighbor : function
            Function to get the neighbor of the solution.

        Returns
        -------
        neighbors : array-like, shape = (n_cities * (n_cities - 1) / 2, n_cities)
            Returns the complete neighborhood of the solution
        """
        neighbors = [[None for i in xrange(self.number_of_cities_)] for j in
                     xrange(self.number_of_cities_ * (self.number_of_cities_ - 1) / 2)]
        idx = 0
        for i in range(self.number_of_cities_):
            for j in range(i + 1, self.number_of_cities_):
                neighbors[idx] = get_neighbor(solution, i, j)
                idx += 1
        return unique(neighbors)

    def prune_tabu_list(self):
        """Prune the tabu list if it exceeds the tabu size.
        """
        while len(self.tabu_) > self.tabu_size:
            self.tabu_.pop(0)

    def update_best_solution(self, solution, cost):
        """Update the best solution attribute.

        Parameters
        ----------
        solution : array-like, shape = (n_cities)
            Solution to set as the best solution found so far.

        cost: float
            Objective function value of the best solution
        """
        self.best_solution_ = solution
        self.best_cost_ = cost

    def _report(self, n_iteration, time_elapsed):
        """Print a summary of the parameters of the solution.

        Parameters
        ----------
        n_iteration : int
            Iteration number at which print the summary

        time_elapsed : float
            Time elapsed since the start of the algorithm

        """
        print "Iteration number", n_iteration
        print "\t * Current solution:", list(self.current_solution)
        print "\t * Current cost:", self.current_cost
        print "\t * Time elpased:", time_elapsed, 'seconds'
        print "\t * Tabu list length:", len(self.tabu_)


