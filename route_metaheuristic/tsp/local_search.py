import time
from route_metaheuristic.tsp.tsp import TSP


class TSPLocalSearch(TSP):
    """Local search on the traveling salesman problem.

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


    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Travelling_salesman_problem

    .. [2] https://en.wikipedia.org/wiki/2-opt

    .. [3] Lenstra, J. K. (1997). Local search in combinatorial
    optimization. Princeton University Press.

    Examples
    --------
    >>> from route_metaheuristic.tsp import TSPLocalSearch
    >>> tsp = TSPLocalSearch(distance_matrix=[[0, 3, 2, 4],
    >>>                                       [3, 0, 2, 3],
    >>>                                       [2, 2, 0, 1],
    >>>                                       [4, 3, 1, 0]])
    >>> tsp.run(max_iter=1000)
    ([3, 4, 2, 1], 9)
    """
    def __init__(self,
                 distance_matrix,
                 initial_solution_strategy='greedy',
                 neighbor_selection='first',
                 neighborhood='2-opt',
                 seed=None):
        TSP.__init__(
            self,
            distance_matrix,
            initial_solution_strategy,
            neighbor_selection,
            neighborhood,
            seed)

    def run(self, max_iter, verbose=None):
        """Run a local search over the algorithm instance.

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
        while number_of_iterations < max_iter:
            try:
                best_neighbor, best_cost = self._select_neighbor(self.current_solution_)
                if best_neighbor is None:
                    break
                self.current_solution_ = best_neighbor
                self.current_cost_ = best_cost
                if verbose and number_of_iterations % verbose == 0 and number_of_iterations != 0:
                    time_elapsed = time.time() - time_init
                    self._report(number_of_iterations, time_elapsed)
                number_of_iterations += 1
            except KeyboardInterrupt:
                break

        time_elapsed = time.time() - time_init
        self._report(number_of_iterations, time_elapsed)

        return self.current_solution_, self.current_cost_

    def _select_neighbor(self, solution):
        """Select neighbor from solution to set as the new solution.

        The strategy to follow is set in the neighbor_selection parameter
        and the neighborhood to explore is set in the neighborhood
        parameter.

        Parameters
        ----------
        solution : array-like, shape = (n_cities)
            Original solution whose neighborhood will be explored.

        Returns
        -------
        best_neighbor : array-like, shape = (n_cities)
            Returns neighbor to return acording to the strategy.

        best_cost : float
            Returns cost of the best_neighbor
        """

        best_neighbor = None
        best_cost = None
        cost_improvement = 0
        for i in range(self.number_of_cities_):
            for j in range(i + 1, self.number_of_cities_):
                neighbor = self.get_neighbor(solution, i, j)
                neighbor_cost = self._evaluate_solution(neighbor)
                if neighbor_cost - self.current_cost_ < cost_improvement:
                    cost_improvement = neighbor_cost - self.current_cost_
                    best_neighbor = neighbor
                    best_cost = neighbor_cost
                    if self.neighbor_selection == 'first':
                        return best_neighbor, best_cost
        return best_neighbor, best_cost

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
