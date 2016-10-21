from nose.tools import assert_equal

from route_metaheuristic.utils import unique


class TestUtils:
    def test_unique(self):
        test_list = [[1, 2], [3, 4], [1, 2]]
        assert_equal(unique(test_list), [[1, 2], [3, 4]])
