import time
from route_metaheuristic.cvrp import CVRP


class CVRPLocalSearch(CVRP):
    """Local search on the Capacitated Vehicle Routing Problem.

    Parameters
    ----------
    distance_matrix : array-like, shape = (n_cities, n_cities)
        Relative distances or costs between cities of the problem.
        Main diagonal elements of the matrix are expected to be 0.

    demand : array-like, shape  = (n_cities)
        Demand of each city. Depot demand is expected to be 0.

    truck_capacitiy: float
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


    Attributes
    ----------
    number_of_cities_ : int
        Number of cities in the problem instance.

    current_solution_: array-like, shape = (n_trucks, n_cities)
        At the moment of the query, the current solution of the problem.
        This attribute can be queried after the creation of the object or
        after the algorithm (run method) has been called.

    References
    ----------

    .. [1] Lenstra, J. K. (1997). Local search in combinatorial
    optimization. Princeton University Press.

    Examples
    --------
    >>> from route_metaheuristic.cvrp import CVRPLocalSearch
    >>> cvrp = CVRPLocalSearch(distance_matrix=[[0, 3, 2, 4], [3, 0, 2, 3], [2, 2, 0, 1], [4, 3, 1, 0]], demand=[0, 2, 1, 2], truck_capacity=3, seed=1)
    >>> cvrp.run(max_iter=1000)
    ([[0, 2, 3], [0, 1]], 13)
    """

    def __init__(self,
                 distance_matrix,
                 demand,
                 truck_capacity,
                 depot=0,
                 initial_solution=None,
                 initial_solution_strategy='random',
                 neighborhood='2-opt-star',
                 seed=None):

        CVRP.__init__(
            self,
            distance_matrix,
            demand,
            truck_capacity,
            depot, initial_solution,
            initial_solution_strategy,
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

        Returns
        -------
        current_solution : array-like, shape=(n_trucks, n_cities)
            Best solution found at current stage

        current_cost : float
            Cost of the best solution
        """
        number_of_iterations = 0
        time_init = time.time()
        while number_of_iterations < max_iter:
            try:
                best_neighbor = self._select_neighbor(self.current_solution_)
                if best_neighbor is None:
                    break
                self.current_solution_ = best_neighbor
                if verbose and number_of_iterations % verbose == 0 and number_of_iterations != 0:
                    time_elapsed = time.time() - time_init
                    self._report(number_of_iterations, time_elapsed)
                number_of_iterations += 1
            except KeyboardInterrupt:
                break

        return self.current_solution_, sum([self._route_cost(route) for route in self.current_solution_])

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
        print "\t * Current solution:", self.current_solution_
        print "\t * Current cost:", self._evaluate_solution(self.current_solution_)
        print "\t * Time elpased:", time_elapsed, 'seconds'

