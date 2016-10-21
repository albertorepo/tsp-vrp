from nose.tools import assert_equal
from nose.tools import assert_raises

from tsp.tsp import TSP


class TestTSP:
    def setup(self):
        self.distance_matrix = [[0, 2, 2, 1], [2, 0, 3, 4], [2, 3, 0, 1], [1, 4, 1, 0]]

    def test_initial_solution(self):
        tsp = TSP(distance_matrix=self.distance_matrix, initial_solution_strategy='greedy')
        assert_equal(len(tsp.current_solution_), tsp.number_of_cities_)
        assert_equal(tsp.current_cost_, 7)

        tsp = TSP(distance_matrix=self.distance_matrix, initial_solution_strategy='random')
        assert_equal(len(tsp.current_solution_), tsp.number_of_cities_)

        assert_raises(AttributeError, TSP, distance_matrix=self.distance_matrix, initial_solution_strategy='error')

    def test_neighborhood_strategy(self):
        assert_raises(AttributeError, TSP, distance_matrix=self.distance_matrix, neighborhood='error')

    def test__neighbor_2_opt(self):
        tsp = TSP(distance_matrix=self.distance_matrix)
        solution = [1, 3, 4, 2]
        assert_equal(tsp._neighbor_2_opt(solution, 2, 3), [1, 4, 3, 2])

    def test__neighbor_swap(self):
        tsp = TSP(distance_matrix=self.distance_matrix)
        solution = [1, 3, 4, 2]
        assert_equal(tsp._neighbor_2_opt(solution, 2, 3), [1, 4, 3, 2])

    def test__evaluate_solution(self):
        tsp = TSP(distance_matrix=self.distance_matrix)
        solution = [1, 3, 4, 2]
        assert_equal(tsp._evaluate_solution(solution), 9)

    def test__distance(self):
        tsp = TSP(distance_matrix=self.distance_matrix)
        assert_equal(tsp._distance(3, 4), 1)
        assert_equal(tsp._distance(2, 2), 0)

    def test__select_neighbor(self):
        tsp = TSP(distance_matrix=self.distance_matrix)
        assert_raises(NotImplementedError, tsp._select_neighbor)

    def test__report(self):
        tsp = TSP(distance_matrix=self.distance_matrix)
        assert_raises(NotImplementedError, tsp._report)

    def test_run(self):
        tsp = TSP(distance_matrix=self.distance_matrix)
        assert_raises(NotImplementedError, tsp.run)
