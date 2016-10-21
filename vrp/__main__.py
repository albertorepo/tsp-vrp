from parser import InstanceParser
from local_search import CVRPLocalSearch
import matplotlib.pyplot as plt

from sa import CVRPSimulatedAnnealing
from tabu_search import CVRPTabuSearch


def main():
    parser = InstanceParser('../A-n32-k5.vrp')
    q, dists, demand = parser.retrieve_instance()
    # cvrp = CVRPSimulatedAnnealing(distance_matrix=dists, demand=demand, truck_capacity=q, depot=0,
    #                             initial_solution_strategy='sequential', t_min=0.01, t_max=50, max_iterations_at_each_t=10)
    cvrp = CVRPTabuSearch(distance_matrix=dists, demand=demand, truck_capacity=q, depot=0,
                          initial_solution_strategy='random', tabu_size=10)
    solution, cost = cvrp.run(max_iter=50, verbose=1)
    print "Cost:", cost
    color = plt.get_cmap('gnuplot2')
    l = color.N / len(solution)
    for (n, k) in enumerate(solution):
        x = [parser.x_coord[i] for i in k]
        y = [parser.y_coord[i] for i in k]

        plt.plot(x, y, color=color(n * l))
        plt.scatter(x, y, color=color(n * l))
        plt.plot([x[0], x[-1]], [y[0], y[-1]], color=color(n * l))

    plt.show()

    cvrp = CVRPLocalSearch(distance_matrix=dists, demand=demand, truck_capacity=q, depot=0, initial_solution=solution,
                           neighborhood='2-opt')
    solution, cost = cvrp.run(max_iter=1000, verbose=5)
    print "Cost:", cost

    color = plt.get_cmap('gnuplot2')
    l = color.N / len(solution)
    for (n, k) in enumerate(solution):
        x = [parser.x_coord[i] for i in k]
        y = [parser.y_coord[i] for i in k]

        plt.plot(x, y, color=color(n * l))
        plt.scatter(x, y, color=color(n * l))
        plt.plot([x[0], x[-1]], [y[0], y[-1]], color=color(n * l))

    plt.show()


if __name__ == '__main__':
    main()
