from local_search import TSPLocalSearch
from sa import TSPSimulatedAnnealing
from tabu_search import TSPTabuSearch
from parser import InstanceParser

import matplotlib.pyplot as plt


def main():
    parser = InstanceParser('../xqf131.txt')
    dists = parser.retrieve_instance()
    tsp_ls = TSPTabuSearch(distance_matrix=dists, tabu_size=10, seed=1)
    sol, cost = tsp_ls.run(max_iter=100, verbose=10)
    print "Solution", sol
    print "Cost", cost

    x_sorted = [parser.x_coord[i - 1] for i in sol]
    y_sorted = [parser.y_coord[i - 1] for i in sol]

    plt.plot(x_sorted, y_sorted)
    plt.scatter(x_sorted, y_sorted)
    plt.plot([x_sorted[0], x_sorted[-1]], [y_sorted[0], y_sorted[-1]], color='blue')
    plt.show()

if __name__ == '__main__':
    main()