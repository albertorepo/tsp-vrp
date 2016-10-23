import time
import math
from cvrp import CVRP
import random
import numpy as np


class CVRPSimulatedAnnealing(CVRP):
    """Simmulated Aeannling on theCapacitated Vehicle Routing Problem.

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

    t_max : float
    Starting (and, therefore, maximum) temperature for the annealing.

    t_min : float
        Temperature at which the algorithm will stop.

    cooling_function : function
        Function that takes as attributes the current temperature and
        the iteration number and returns a new temperature.

        - Linear:           (lambda (t, i): self.t_max - i * beta)
        - Geometric:        (lambda (t, i): alpha * t)
        - Logarithmic:      (lambda (t, i): self.t_max / log(i))
        - Slow decrease:    (lambda (t, i): t / (1 + beta * t)

    max_iterations_at_each_t : int
        Number of iterations to look for a new solution without change
        the temperature.


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

    .. [1] Osman, I. H. (1993). Metastrategy simulated annealing and tabu
    search algorithms for the vehicle routing problem. Annals of operations
    research, 41(4), 421-451.

    Examples
    --------
    >>> from route_metaheuristic.cvrp import CVRPSimulatedAnnealing
    >>> cvrp = CVRPSimulatedAnnealing(distance_matrix=[[0, 3, 2, 4], [3, 0, 2, 3], [2, 2, 0, 1], [4, 3, 1, 0]], demand=[0, 2, 1, 2], truck_capacity=3, seed=1)
    >>> cvrp.run(max_iter=1000)
    ([[0, 1], [0, 3, 2]], 13)
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
                 t_max=100,
                 t_min=0.01,
                 cooling_function=(lambda (t, i): t * 0.99),
                 max_iterations_at_each_t=50):
        CVRP.__init__(
            self,
            distance_matrix,
            demand,
            truck_capacity,
            depot, initial_solution,
            initial_solution_strategy,
            neighborhood,
            seed)
        self.t_max = float(t_max)
        self.t_min = float(t_min)
        self.cooling_function = cooling_function
        self.max_iterations_at_each_t = max_iterations_at_each_t

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
                        self.current_solution_ = random_neighbor
                iterations += 1
            if verbose and number_of_iterations % verbose == 0 and number_of_iterations != 0:
                time_elapsed = time.time() - time_init
                self._report(number_of_iterations, time_elapsed, t)
            number_of_iterations += 1
            t = self.cooling_function((t, iterations))

        return self.current_solution_, self._evaluate_solution(self.current_solution_)

    def _get_random_feasible_neighbor(self):
        """Get a random and feasible neighbor among the neighborhood
        of the current solution.

        The neighborhood used is defined in the neighborhood object
        parameter.


        Returns
        -------
        neighbor : array-like, shape=(n_trucks, n_cities)
            Returns a random and feasible neighbor
        """
        while True:
            route1, route2 = random.sample(self.current_solution_, 2)
            u = random.randint(0, len(route1) - 1)
            v = random.randint(0, len(route2) - 1)
            if self._2opt_change_is_feasible(route1, route2, u, v):
                break
        r1_n, r2_n = self.current_solution_.index(route1), self.current_solution_.index(route2)
        delta_e = self._increment_of_cost_between_routes(route1, route2, u, v)
        return self._neighbor_2_opt_star(self.current_solution_, route1, route2, r1_n, r2_n, u, v), delta_e

    def _report(self, n_iteration, time_elapsed, t):
        """Print a summary of the parameters of the solution.

        Parameters
        ----------
        n_iteration : int
            Iteration number at which print the summary

        time_elapsed : float
            Time elapsed since the start of the algorithm

        t : float
            Current temperature of the annealing.
        """
        print "Iteration number", n_iteration
        print "\t * Current solution:", self.current_solution_
        print "\t * Current cost:", self._evaluate_solution(self.current_solution_)
        print "\t * Current temperature:", t
        print "\t * Time elapsed: %.2f seconds" % time_elapsed

