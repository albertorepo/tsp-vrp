import random
from copy import deepcopy

import numpy as np

from route_metaheuristic.basic import Problem


class TSP(Problem):
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

    """

    def __init__(self, distance_matrix,
                 initial_solution_strategy='greedy',
                 neighbor_selection='first',
                 neighborhood='2-opt',
                 seed=None):
        self.distance_matrix = distance_matrix
        self.number_of_cities_ = len(self.distance_matrix)
        self.seed = seed
        self.initial_solution_strategy = initial_solution_strategy
        self.current_solution_ = self._get_initial_solution()
        self.current_cost_ = self._evaluate_solution(self.current_solution_)
        self.neighbor_selection = neighbor_selection
        self.neighborhood = neighborhood

        if self.neighborhood == '2-opt':
            self.get_neighbor = self._neighbor_2_opt
        elif self.neighborhood == 'swap':
            self.get_neighbor = self._neighbor_swap
        else:
            raise AttributeError('`neighborhood` must be either `2-opt` or `swap`')

    def _get_initial_solution(self):
        """Generate a starting point to solve the problem.

        Returns
        -------
        solution: array-like of shape = (n_cities)
            A permutation that constitutes a solution of the TSP.
        """

        if self.initial_solution_strategy == 'random':
            solution = range(1, self.number_of_cities_ + 1)
            random.seed(self.seed)
            random.shuffle(solution)
            return solution
        elif self.initial_solution_strategy == 'greedy':
            bag_of_cities = range(0, self.number_of_cities_)
            random.seed(self.seed)
            solution = [random.randint(1, self.number_of_cities_)]
            bag_of_cities.remove(solution[0] - 1)
            for i in range(1, self.number_of_cities_):
                distances = [self.distance_matrix[solution[i - 1] - 1][j] for j in bag_of_cities]
                solution.append(bag_of_cities[np.argmin(distances)] + 1)
                bag_of_cities.remove(solution[i] - 1)
            return solution
        else:
            raise AttributeError('`initial_solution_strategy` must be either `greedy` or `random`')

    def _neighbor_2_opt(self, solution, i, j):
        """Build a neighbor following the 2-opt neighborhood strategy.

        Parameters
        ----------
        solution: array-like of shape = (n_cities)
            The permutation to be transformed into a new solution.

        i : int
            Starting index to perform the flip.

        j : int
            Ending index to perform the flip

        Returns
        -------
        solution_tmp : array-like of shape = (n_cities)
            Return the generated neighbor.
        """

        solution_tmp = deepcopy(solution)
        solution_tmp[i - 1:j] = list(reversed(solution[i - 1:j]))
        return solution_tmp

    def _neighbor_swap(self, solution, i, j):
        """Build a neighbor following the swap neighborhood strategy.

         Parameters
         ----------
         solution: array-like of shape = (n_cities)
            The permutation to be transformed into a new solution.

        i : int
            First index to perform the swap.

        j : int
            Second index to perform the swap.

        Returns
        -------
        solution_tmp : array-like of shape = [n_cities]
            Return the generated neighbor.
        """

        solution_tmp = deepcopy(solution)
        solution_tmp[i], solution_tmp[j] = solution_tmp[j], solution_tmp[i]
        return solution_tmp

    def _evaluate_solution(self, solution, idx=1):
        """Calculate the objective function value of a given solution
        in a recursive way.

        The objective function will be the sum of the distances of every
        consecutive city on the permutation plus the distance between the
        last one and the first one.

        Parameters
        ----------
        solution : array-like of shape = (n_cities)
            The permutation to be evaluated.

        idx : int, optional (default=1)
            Index of the city to calculate the distance and add it to
            the recursive solution. When idx hits the lenght of the
            permutation, the distance between first and last cities
            is add the total distance.

        Returns
        -------
        distance: float
            Returns the cost of the permutation.
        """
        if idx == len(solution):
            return self._distance(solution[idx - 1], solution[0])
        else:
            return self._distance(solution[idx], solution[idx - 1]) + self._evaluate_solution(solution, idx=idx + 1)

    def _distance(self, a, b):
        """Get the distance between two cities

        Parameters
        ----------
        a : int
            Number (not index) of the first city.

        b : int
            Number (not index) of the second city.

        Returns
        -------
        distance : float
            Return distance between city a and b.
        """

        return self.distance_matrix[a - 1][b - 1]

    def _select_neighbor(self, *args, **kwargs):
        raise NotImplementedError('`_select_neighbor` has not been implemented for this algorithm.')

    def _report(self, *args, **kwargs):
        raise NotImplementedError('`_report` has not been implemented for this algorithm.')

    def run(self, *args, **kwargs):
        raise NotImplementedError('`run` has not been implemented for this algorithm.')
