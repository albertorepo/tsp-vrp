from nose.tools import assert_equal

from tsp.tsp import TSP


class TestTSP:

    def setup(self):
        distance_matrix = [[0, 2, 2, 1], [2, 0, 3, 4], [2, 3, 0, 1], [1, 4, 1, 0]]
        self.tsp = TSP(distance_matrix=distance_matrix)

    def test_evaluate_solution(self):

        solution = [1, 3, 4, 2]
        assert_equal(self.tsp._evaluate_solution(solution), 9)

    def test_initial_solution(self):
        assert_equal(self.tsp.current_cost, 7)