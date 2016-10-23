import time
from copy import deepcopy
import numpy as np
from cvrp import CVRP


class CVRPTabuSearch(CVRP):
    """Tabu search on the Capacitated Vehicle Routing Problem.

    Parameters
    ----------
    distance_matrix : array-like, shape = (n_cities, n_cities)
        Relative distances or costs between cities of the problem.
        Main diagonal elements of the matrix are expected to be 0.

    demand : array-like, shape  = (n_cities)
        Demand of each city. Depot demand is expected to be 0.

    truck_capacity: float
        Capacitiy of every truck.

    depot : int, optional, (default=0)
        Index of the depot on distance_matrix and demand parameters.

    initial_solution_strategy : string, optional, (default='random')
        Strategy for creating an initial solution. Supported strategies
        are 'sequential' for a sequential constructional solution and
        'random' for a random generated solution.

    neighborhood : string, optional, (default='2-opt-star')
        Neighborhood to explore at each iteration of the algorithm. Supported
        neighborhoods are '2-opt-star' for the 2-opt* inter-neighborhood and
        '2-opt' for the 2-opt intra-neighborhood.

    seed : int, optional, (default=None)
        Random seed for every randomized choice.

    tabu_size : int
        Size of the tabu list. Maximum number of solutions to forbid.


    Attributes
    ----------
    number_of_cities_ : int
        Number of cities in the problem instance.

    current_solution_: array-like, shape = (n_trucks, n_cities)
        At the moment of the query, the current solution of the problem.
        This attribute can be queried after the creation of the object or
        after the algorithm (run method) has been called.

    current_cost_ : float
        Cost of the current solution

    tabu_ : array-like, shape = (tabu_size, n_trucks, n_cities)
        Actual tabu list. Set of solutions that are forbidden.

    best_solution_ : array-like, shape = (n_trucks, n_cities)
        Best solution found so far.

    best_cost_ : float
        Objective function value of the best solution.

    References
    ----------

    .. [1] Gendreau, M., Hertz, A., & Laporte, G. (1994). A tabu search
    heuristic for the vehicle routing problem. Management science, 40(10),
    1276-1290.

    Examples
    --------
    >>> from route_metaheuristic.cvrp import CVRPTabuSearch
    >>> cvrp = CVRPTabuSearch(distance_matrix=[[0, 3, 2, 4], [3, 0, 2, 3], [2, 2, 0, 1], [4, 3, 1, 0]], demand=[0, 2, 1, 2], truck_capacity=3, seed=1, tabu_size=1)
    >>> cvrp.run(max_iter=1000)
    ([[0, 3], [0, 2], [0, 1]], 18)
    """

    def __init__(self,
                 distance_matrix,
                 demand,
                 truck_capacity,
                 depot=0,
                 initial_solution=None,
                 initial_solution_strategy='random',
                 neighborhood='2-opt-star',
                 seed=None,
                 tabu_size=None):
        CVRP.__init__(
            self,
            distance_matrix,
            demand,
            truck_capacity,
            depot,
            initial_solution,
            initial_solution_strategy,
            neighborhood,
            seed)
        self.tabu_size = tabu_size
        self.tabu = [self.current_solution_]
        self.current_cost_ = self._evaluate_solution(self.current_solution_)
        self.best_solution = self.current_solution_
        self.best_cost = self.current_cost_

    def run(self, max_iter, verbose=None):
        """Run a tabu search over the algorithm instance.

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
            best_neighbor, best_neighbor_cost = self._select_neighbor(self.current_solution_)
            if best_neighbor is None:
                break
            self.current_solution_ = best_neighbor
            self.current_cost_ = best_neighbor_cost

            if best_neighbor_cost < self.best_cost:
                self.update_best_solution(best_neighbor, best_neighbor_cost)

            if verbose and number_of_iterations % verbose == 0 and number_of_iterations != 0:
                time_elapsed = time.time() - time_init
                self._report(number_of_iterations, time_elapsed)

            number_of_iterations += 1
            self.tabu.append(self.current_solution_)

        return self.best_solution, self.best_cost

    def _select_neighbor(self, solution):
        """Select neighbor from solution to set as the new solution.

        Tabu list will be checked in order to ensure that the
        neighbor is not in it.

        Parameters
        ----------
        solution : array-like, shape = (n_trucks, n_cities)
            Original solution whose neighborhood will be explored.

        Returns
        -------
        best_neighbor : array-like, shape = (n_trucks, n_cities)
            Returns neighbor to return.
        best_cost : float
            Returns cost of the best_neighbor
        """
        neighbors = self._neighborhood(solution)
        if not neighbors:
            return None, None
        tweaked_neighbors = [neighbor for neighbor in neighbors if neighbor not in self.tabu]
        costs = [self._evaluate_solution(neighbor) for neighbor in tweaked_neighbors]
        best_neighbor, best_neighbor_cost = neighbors[np.argmin(costs)], np.min(costs)
        return best_neighbor, best_neighbor_cost

    def _neighborhood(self, solution):
        """Get the neighborhood of a solution based on an interchange of nodes.

        Parameters
        ----------
        solution : array-like, shape = (n_trucks, n_cities)
            Original solution whose neighborhood will be explored.

        Returns
        -------
        neighbors : array-like, shape = (n_neighbors, n_trucks, n_cities)
            Returns the complete neighborhood of the solution
        """
        neighbors = []
        for r in range(len(solution)):
            for t in range(len(solution)):
                if r == t:
                    continue
                for i in range(1, len(solution[r])):
                    for j in range(1, len(solution[t])):
                        neighbor = deepcopy(solution)
                        neighbor[t].insert(j, neighbor[r][i])
                        del neighbor[r][i]
                        if self._solution_is_feasible(neighbor):
                            neighbors.append(neighbor)
        return [route for route in neighbors if len(route) > 1]

    def prune_tabu_list(self):
        """Prune the tabu list if it exceeds the tabu size.
        """
        while len(self.tabu) > self.tabu_size:
            self.tabu.pop(0)

    def update_best_solution(self, solution, cost):
        """Update the best solution attribute.

        Parameters
        ----------
        solution : array-like, shape = (n_trucks, n_cities)
            Solution to set as the best solution found so far.

        cost: float
            Objective function value of the best solution
        """
        self.best_solution = solution
        self.best_cost = cost

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
        print "\t * Current solution:", list(self.current_solution_)
        print "\t * Current cost:", self.current_cost_
        print "\t * Time elpased:", time_elapsed, 'seconds'
        print "\t * Tabu list length:", len(self.tabu)

    def _solution_is_feasible(self, solution):
        """Whether or not a given solution respect the load constraints.

        Parameters
        ----------

        solution : array-like, shape=(n_trucks, n_cities)
            Possible solution to check.

        Returns
        -------
        feasible :
            Returns whether or not the solution is valid.
        """
        demands = [self._load(route) for route in solution]
        return all(demand < self.truck_capacity for demand in demands)
