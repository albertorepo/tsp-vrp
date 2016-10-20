from nose.tools import assert_equals

from vrp.cvrp import CVRP


class TestCVRP:

    def setup(self):
        d = [0, 2, 3, 1, 1]
        dists = [[0, 2, 4, 3, 1],
                 [2, 0, 5, 1, 3],
                 [4, 5, 0, 4, 5],
                 [3, 1, 4, 0, 4],
                 [1, 3, 5, 4, 0]]
        self.cvrp = CVRP(demand=d, distance_matrix=dists, truck_capacity=3, depot=0)


    def test__load(self):
        assert_equals(self.cvrp._load([0, 3, 2, 4, 1]), 7)
        assert_equals(self.cvrp._load([0]), 0)

    def test__route_cost(self):
        assert_equals(self.cvrp._route_cost([0, 3, 2, 4, 1]), 17)

    def test__inter_route_cost(self):
        assert_equals(self.cvrp._inter_route_cost([0,3,1,2,4], [0,1,4,2,3], 3, 5), 4)
