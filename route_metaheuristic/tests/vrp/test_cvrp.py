from nose.tools import assert_equal
from nose.tools import assert_equals
from nose.tools import assert_false
from nose.tools import assert_raises

from route_metaheuristic.vrp.cvrp import CVRP


class TestCVRP:

    def setup(self):
        self.d = [0, 2, 3, 1, 1]
        self.dists = [[0, 2, 4, 3, 1],
                 [2, 0, 5, 1, 3],
                 [4, 5, 0, 4, 5],
                 [3, 1, 4, 0, 4],
                 [1, 3, 5, 4, 0]]



    def test_initial_solution(self):
        cvrp = CVRP(demand=self.d, distance_matrix=self.dists, truck_capacity=3, depot=0,
                    initial_solution_strategy='sequential')
        assert_equal(sum([len(route) for route in cvrp.current_solution]) - len(cvrp.current_solution),
                     cvrp.number_of_cities - 1)
        assert_equal(cvrp._evaluate_solution(cvrp.current_solution), 20)

        cvrp = CVRP(demand=self.d, distance_matrix=self.dists, truck_capacity=3, depot=0,
                    initial_solution_strategy='random')
        assert_equal(sum([len(route) for route in cvrp.current_solution]) - len(cvrp.current_solution),
                     cvrp.number_of_cities - 1)

        assert_raises(AttributeError, CVRP, demand=self.d, distance_matrix=self.dists, truck_capacity=3, depot=0,
                    initial_solution_strategy='error')

    def test__load(self):
        cvrp = CVRP(demand=self.d, distance_matrix=self.dists, truck_capacity=3, depot=0)
        assert_equals(cvrp._load([0, 3, 2, 4, 1]), 7)
        assert_equals(cvrp._load([0]), 0)

    def test__route_cost(self):
        cvrp = CVRP(demand=self.d, distance_matrix=self.dists, truck_capacity=3, depot=0)
        assert_equals(cvrp._route_cost([0, 3, 2, 4, 1]), 17)

    def test__inter_route_cost(self):
        cvrp = CVRP(demand=self.d, distance_matrix=self.dists, truck_capacity=3, depot=0)
        assert_equals(cvrp._inter_route_cost([0,3,1,2,4], [0,1,4,2,3], 3, 5), 4)

    def test__neighbor_2_opt_star(self):
        cvrp = CVRP(demand=self.d, distance_matrix=self.dists, truck_capacity=3, depot=0)
        assert_equal(cvrp._neighbor_2_opt_star([[0, 3], [0, 4], [0, 2], [0, 1]], [0,3], [0,4], 0, 1, 0, 1),
                     [[0, 4, 3], [0, 2], [0, 1]])

    def test__neighbor_2_opt(self):
        cvrp = CVRP(demand=self.d, distance_matrix=self.dists, truck_capacity=3, depot=0)
        assert_equal(cvrp._neighbor_2_opt([0, 3, 2, 4, 1], 2, 4), [0, 4, 2, 3, 1])

    def test__increment_of_cost_between_routes(self):
        cvrp = CVRP(demand=self.d, distance_matrix=self.dists, truck_capacity=3, depot=0)
        assert_equal(cvrp._increment_of_cost_between_routes([0, 3, 2, 4, 1], [0,1,4,2,3], 1, 3), -8)

    def test__change_is_feasible(self):
        cvrp = CVRP(demand=self.d, distance_matrix=self.dists, truck_capacity=3, depot=0)
        assert_false(cvrp._2opt_change_is_feasible([0, 3, 2, 4, 1], [0, 1, 4, 2, 3], 1, 3))