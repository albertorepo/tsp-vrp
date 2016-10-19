import random

from basic import Problem
import math


class CVRP(Problem):

    def __init__(self, distance_matrix, demand, truck_capacity, depot=0):
        self.distance_matrix = distance_matrix
        self.number_of_cities = len(self.distance_matrix)
        self.demand = demand
        self.truck_capacity = truck_capacity
        self.depot = depot
        self.current_solution = self._get_initial_solution()

    def _get_initial_solution(self, *args, **kwargs):
        """Sequential route"""
        max_cities_per_truck = int(math.ceil(self.truck_capacity / float(max(self.demand))))
        assert max_cities_per_truck * self.number_of_trucks >= (self.number_of_cities - 1)
        initial_route = [self.depot]
        cities = range(1, self.number_of_cities)
        random.shuffle(cities)
        initial_route.append(cities)

        solution = [[0] for _ in xrange(self.number_of_trucks)]
        truck = 0
        while cities:
            if len(solution[truck]) == (max_cities_per_truck + 1):
                continue
            random_n_cities = random.randint(1, max_cities_per_truck - len(solution[truck]) + 1)
            if random_n_cities > len(cities):
                random_n_cities = len(cities)
            solution[truck] += cities[0:random_n_cities]
            cities = cities[random_n_cities:]
            if truck == self.number_of_trucks - 1:
                truck = 0
            else:
                truck += 1
        return solution

    def _neighborhood(self, *args, **kwargs):
        pass

    def _evaluate_solution(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    dists = [[0,8,9,6,2,1],
             [9,0,3,4,6,1],
             [9,5,0,6,8,7],
             [5,3,1,0,8,7],
             [9,8,6,2,0,4],
             [7,8,3,1,5,0]]
    cvrp = CVRP(distance_matrix=dists, number_of_trucks=3, demand=[0,3,2,3,2,3],truck_capacity=4,depot=0)
    print cvrp.current_solution