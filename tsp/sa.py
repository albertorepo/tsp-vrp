import time

from tsp import TSP
import numpy as np


class TSPSimulatedAnnealing(TSP):
    """Simulated annealing on the traveling salesman problem.

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

    -- [3] Kirkpatrick, S. (1984). Optimization by simulated annealing:
    Quantitative studies. Journal of statistical physics, 34(5-6), 975-986.

    """

    def __init__(self,
                 distance_matrix,
                 initial_solution_strategy='greedy',
                 neighbor_selection='first',
                 neighborhood='2-opt',
                 seed=None,
                 t_max=100,
                 t_min=0.01,
                 cooling_function=(lambda (t, i): t * 0.99),
                 max_iterations_at_each_t=50):
        TSP.__init__(
            self,
            distance_matrix,
            initial_solution_strategy,
            neighbor_selection,
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
                random_neighbor, random_neighbor_cost = self._select_neighbor(self.current_solution)
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

    def _select_neighbor(self, solution):
        """Select a random neighbor from solution to set as the new solution.

        Parameters
        ----------
        solution : array-like, shape = (n_cities)
            Original solution whose neighborhood will be explored.

        Returns
        -------
        random_neighbor : array-like, shape = (n_cities)
            Returns neighbor to return.

        random_neighbor_cost : float
            Returns cost of the random_neighbor
        """

        i = np.random.randint(0, self.number_of_cities_ - 1)
        j = np.random.randint(i + 1, self.number_of_cities_)
        random_neighbor = self.get_neighbor(self.current_solution, i, j)
        random_neighbor_cost = self._evaluate_solution(random_neighbor)
        return random_neighbor, random_neighbor_cost

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
        print "\t * Current solution:", list(self.current_solution)
        print "\t * Current cost:", self.current_cost
        print "\t * Current temperature:", t
        print "\t * Time elapsed: %.2f seconds" % time_elapsed


