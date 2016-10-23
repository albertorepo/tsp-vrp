import itertools
import math
import random
from copy import deepcopy

import numpy as np

from route_metaheuristic.basic import Problem


class CVRP(Problem):
    """Capacitated Vehicle Routing Problem.

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

    .. [1] S.N. Kumar, R. Panneerselvam (2012). A Survey on the Vehicle Routing
    Problem and Its Variants. Intelligent Information Management, 4-3, 66-74
    """
    def __init__(self, distance_matrix, demand, truck_capacity, depot=0, initial_solution=None,
                 initial_solution_strategy='random', neighborhood='2-opt-star', seed=None):
        self.distance_matrix = distance_matrix
        self.number_of_cities_ = len(self.distance_matrix)
        self.demand = demand
        self.truck_capacity = truck_capacity
        self.depot = depot
        self.initial_solution_strategy = initial_solution_strategy
        self.seed = seed
        if initial_solution:
            self.current_solution_ = initial_solution
        else:
            self.current_solution_ = self._get_initial_solution()
        self.neighborhood = neighborhood


    def _get_initial_solution(self):
        """Generate a starting point to solve the problem.

        Returns
        -------
        solution: array-like of shape = (n_trucks, n_cities)
            An array that constitutes a solution of the CVRP.
        """

        if self.initial_solution_strategy == 'sequential':
            return self._seq_initial_solution()
        elif self.initial_solution_strategy == 'random':
            return self._rand_initial_solution()
        else:
            raise AttributeError('`initial_solution_strategy` must be either `sequential` or `random`')

    def _rand_initial_solution(self):
        """Generate a random solution of the problem.

        Random cities are assigned to a truck until it is full.

        Returns
        -------
        solution: array-like of shape = (n_trucks, n_cities)
            An array that constitutes a solution of the CVRP.
        """
        cities = range(1, self.number_of_cities_)
        max_cities = int(math.ceil(self.truck_capacity / float(max(self.demand))))

        random.seed(self.seed)
        random.shuffle(cities)
        solution = []
        while cities:
            random.seed(self.seed)
            random_n_cities = random.randint(1, max_cities)
            solution.append([0] + cities[0:random_n_cities])
            cities = cities[random_n_cities:]

        return solution

    def _seq_initial_solution(self):
        """Generate a sequential built solution of the problem.

        Starting from the depot, a route is progressively extended
        by adding the nearest unrouted customer, among those compatible
        with the vehicle residual capacity.

        Returns
        -------
        solution: array-like of shape = (n_trucks, n_cities)
            An array that constitutes a solution of the CVRP.
        """
        cities = range(1, self.number_of_cities_)
        solution = []
        while cities:
            solution.append([0])
            while cities:
                demand = sum([self.demand[i] for i in solution[-1]])
                distances_of_last_city = self.distance_matrix[solution[-1][-1]]
                distances_of_feasible_cities = [distances_of_last_city[i] for i in cities]
                nearest_city = cities[np.argmin(distances_of_feasible_cities)]
                if demand + self.demand[nearest_city] > self.truck_capacity:
                    break
                solution[-1].append(nearest_city)
                cities.remove(nearest_city)

        return solution

    def _select_neighbor(self, solution):
        """Select neighbor from solution to set as the new solution.

        The neighborhood to explore is set in the neighborhood
        parameter.

        Parameters
        ----------
        solution : array-like, shape = (n_trucks, n_cities)
            Original solution whose neighborhood will be explored.

        Returns
        -------
        best_neighbor : array-like, shape = (n_trucks, n_cities)
            Returns the best feasible neighbor.
            If there isn't any feasible solution among the neighborhood,
            it returns None.
        """
        if self.neighborhood == '2-opt-star':
            return self._neighborhood_2_opt_star(solution)
        elif self.neighborhood == '2-opt':
            return self._neighborhood_2_opt(solution)
        else:
            raise AttributeError('`neighborhood` must be either `2-opt-star` or `2-opt`')

    def _neighborhood_2_opt_star(self, solution):
        """Get the best solution in the 2-opt* neighborhood of the
        solution.

        Parameters
        ----------
        solution : array-like, shape = (n_trucks, n_cities)
            Original solution whose neighborhood will be explored.

        Returns
        -------
        best_neighbor : array-like, shape = (n_trucks, n_cities)
            Returns the best feasible neighbor.
            If there isn't any feasible solution among the neighborhood,
            it returns None.
        """
        solution_tmp = deepcopy(solution)
        best_incremental_cost = 0
        for (r_n, R), (t_n, T) in itertools.combinations(enumerate(solution_tmp), 2):
            for u in range(0, len(R)):
                for v in range(0, len(T)):
                    if self._2opt_change_is_feasible(R, T, u, v):
                        incremental_cost = self._increment_of_cost_between_routes(R, T, u, v)
                        if incremental_cost < best_incremental_cost:
                            best_incremental_cost = incremental_cost
                            best_u = u
                            best_v = v
            if best_incremental_cost < 0:
                return self._neighbor_2_opt_star(solution_tmp, R, T, r_n, t_n, best_u, best_v)
        return None

    def _neighbor_2_opt_star(self, solution, route1, route2, n1, n2, u, v):
        """Build a neighbor following the 2-opt* neighborhood strategy.

        Parameters
        ----------
        solution: array-like of shape = (n_trucks, n_cities)
            The permutation to be transformed into a new solution.

        route1 : array-like, shape=(n_cities)
            First route to be transformed.

        route2 : array-like, shape=(n_cities)
            Second route to be transformed.

        n1 : int
            Index of route1 in solution

        n2 : int
            Index of route2 in solution

        u : int
            Index of the node in route1 to do the exchange.

        v : int
            Index of the node in route2 to do the exchange

        Returns
        -------
        solution_tmp : array-like of shape = (n_trucks, n_cities)
            Return the generated neighbor.
        """

        route1_tmp = route1[0:u + 1] + route2[v + 1:]
        route2_tmp = route2[0:v + 1] + route1[u + 1:]
        solution[n1] = route1_tmp
        solution[n2] = route2_tmp
        return [route for route in solution if len(route) > 1]

    def _neighborhood_2_opt(self, solution):
        """Get the best solution in the 2-opt neighborhood of the
        solution.

        Parameters
        ----------
        solution : array-like, shape = (n_trucks, n_cities)
            Original solution whose neighborhood will be explored.

        Returns
        -------
        best_neighbor : array-like, shape = (n_cities)
            Returns the best feasible neighbor.
            If there isn't any feasible solution among the neighborhood,
            it returns None.
        """
        solution_tmp = deepcopy(solution)
        for n, route in enumerate(solution_tmp):
            for i in range(1, len(route)):
                for j in range(i + 1, len(route)):
                    neighbor = self._neighbor_2_opt(route, i, j)
                    if self._route_cost(neighbor) < self._route_cost(route):
                        solution_tmp[n] = neighbor
                        return solution_tmp
        return None

    def _neighbor_2_opt(self, route, i, j):
        """Build a neighbor following the 2-opt neighborhood strategy.

        Parameters
        ----------
        route: array-like of shape = (n_cities)
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
        solution_tmp = deepcopy(route)
        solution_tmp[i - 1:j] = list(reversed(route[i - 1:j]))
        return solution_tmp

    def _load(self, route, i=0, j=None):
        """Calculate the total load of a route or partial route
        in terms of the demand of the cities on it.

        If neither starting nor ending indexes are provided,
        entire route demand is returned.

        Parameters
        ----------

        route : array-like, shape=(n_cities)
            Route whose load will be calculated. It is a possible
            solution of the problem.

        i : int, optional, (default=0)
            If a partial load is required, the starting index of
            the sub-route.

        j : int, optional, (default=None)
            If a partial load is required, the ending index of
            the sub-route.

        Returns
        -------
        load : float
            Total load of the sub-route.

        """
        if not j:
            j = len(route) - 1

        return sum([self.demand[route[k]] for k in range(i, j + 1)])

    def _route_cost(self, route, i=0, j=None):
        """Calculate the of a route or partial route
        in terms of the distance between cities on it.

        If neither starting nor ending indexes are provided,
        entire route cost is returned.

        Parameters
        ----------

        route : array-like, shape=(n_cities)
            Route whose cost will be calculated. It is a possible
            solution of the problem.

        i : int, optional, (default=0)
            If a partial cost is required, the starting index of
            the sub-route.

        j : int, optional, (default=None)
            If a partial cost is required, the ending index of
            the sub-route.

        Returns
        -------
        dist : float
            Total cost of the sub-route.

        """
        if not j:
            j = len(route)
        dist = 0
        for k in range(i, j):
            if k == len(route) - 1:
                dist += self.distance_matrix[route[k]][route[0]]
            else:
                dist += self.distance_matrix[route[k]][route[k + 1]]
        return dist

    def _inter_route_cost(self, route1, route2, node1, node2):
        """Calculate the cost between two nodes on different routes.

        Parameters
        ----------
        route1 : array-like, shape=(n_cities)
            First route.

        route2 : array-like, shape=(n_cities)
            Second route.

        node1 : int
            Index of the node in the first route.

        node2 : int
            Index of the node in the second route.

        Returns
        -------
        distance : float
            Distance between the two nodes.
        """
        if node1 == len(route1):
            node1 = 0
        if node2 == len(route2):
            node2 = 0
        return self.distance_matrix[route1[node1]][route2[node2]]

    def _increment_of_cost_between_routes(self, route1, route2, u, v):
        """Calculate the change on the objective function if a 2-opt*
        change is performed.

        Parameters
        ----------
        route1 : array-like, shape=(n_cities)
            First route to be transformed.

        route2 : array-like, shape=(n_cities)
            Second route to be transformed.

        u : int
            Index of the node in route1 to do the exchange.

        v : int
            Index of the node in route2 to do the exchange

        Returns
        -------
        delta : float
            Increment (or decrement) of cost between the two routes.
        """
        return self._inter_route_cost(route1, route2, u, v + 1) + \
               self._inter_route_cost(route2, route1, v, u + 1) - \
               self._route_cost(route1, u, u + 1) - \
               self._route_cost(route2, v, v + 1)

    def _2opt_change_is_feasible(self, route1, route2, u, v):
        """Whether or not a 2-opt* change is feasible in terms of
        the total demand of the resulting routes and the capacity
        of the trucks.

        Parameters
        ----------
        route1 : array-like, shape=(n_cities)
            First route to be transformed.

        route2 : array-like, shape=(n_cities)
            Second route to be transformed.

        u : int
            Index of the node in route1 to do the exchange.

        v : int
            Index of the node in route2 to do the exchange

        Returns
        -------
        feasible : bool
            Returns whether or not the change is feasible
        """
        load_1_until_u = sum(self.demand[route1[k]] for k in range(u + 1))
        load_2_until_v = sum(self.demand[route2[k]] for k in range(v + 1))
        return load_1_until_u + self._load(route2) - load_2_until_v <= self.truck_capacity and \
               load_2_until_v + self._load(route1) - load_1_until_u <= self.truck_capacity

    def _evaluate_solution(self, solution):
        """Calculate the objective function value of a given solution.

        The objective function will be the sum of the costs of every route
        in the solution.

        Parameters
        ----------
        solution : array-like, shape = (n_trucks, n_cities)
            Solution whose cost will be estimated.

        Returns
        -------
        distance: float
            Returns the cost of the solution.
        """
        return sum([self._route_cost(route) for route in solution])

    def run(self, *args, **kwargs):
        raise NotImplementedError
