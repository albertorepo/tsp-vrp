import math

from vrp.cvrp import CVRP


class InstanceParser:

    def _init__(self, input_file, number_of_trucks):
        self.input_file = input_file
        self.number_of_trucks = number_of_trucks

    def retrieve_instance(self):

        file_iter = self._read_file()
        while True:
            line = file_iter.next()
            if line == 'DIMENSION':
                file_iter.next()
                number_of_nodes = int(file_iter.next())
                number_of_costumers = number_of_nodes - 1
            elif line == 'CAPACITY':
                file_iter.next()
                truck_capacity = int(file_iter.next())
            elif line == 'EDGE_WEIGHT_TYPE':
                file_iter.next()
                line = file_iter.next()
                if line == 'EUC_2D':
                    raise AttributeError('EDGE_WEIGHT_TYPE unsuported: %s' % line)
            elif line == 'NODE_COORD_SECTION':
                break

        x_coord = [None] * number_of_nodes
        y_coord = [None] * number_of_nodes

        for node in range(number_of_nodes):
            node_id = int(file_iter.next())
            if node_id != node + 1:
                raise AttributeError('Uncontinous index of nodes in coordinates section')
            x_coord[node] = int(file_iter.next())
            y_coord[node] = int(file_iter.next())

        distance_matrix  = self._calculate_distance_matrix(x_coord, y_coord)

        line = file_iter.next()
        if line != 'DEMAND_SECTION':
            raise AttributeError('DEMAND_SECTION not present in input file')

        demand = [None] * number_of_nodes
        for node in range(number_of_nodes):
            node_id = int(file_iter.next())
            if node_id != node + 1:
                raise AttributeError('Uncontinous index of nodes in demand section')
            demand[node] = int(file_iter.next())

        line = file_iter.next()
        if line != 'DEPOT_SECTION':
            raise AttributeError('DEPOT_SECTION not present in input file')

        warehouse_id = file_iter.next()
        if warehouse_id != 1:
            raise AttributeError('Warehouse ID is supossed to be 1')

        end_of_depot_section = file_iter.next()
        if end_of_depot_section != -1:
            raise AttributeError('Only one warehouse expected, more than one found')

        if demand[0] != 0:
            raise AttributeError('Warehouse demand is suposed to be 0')

        return CVRP(number_of_costumers, truck_capacity, distance_matrix, demand)

    def _read_file(self):
        with open(self.input_file) as f:
            for line in f.read().split():
                yield line

    def _calculate_distance_matrix(self, x_coord, y_coord):
        number_of_nodes = len(x_coord)
        distance_matrix = [[None for i in xrange(number_of_nodes)] for y in xrange(number_of_nodes)]

        for i in range(number_of_nodes):
            for j in range(number_of_nodes):
                distance = self._calculate_distance(x_coord[i], y_coord[i], x_coord[j], y_coord[j])
                distance_matrix[i][j] = distance
        return distance_matrix

    def _calculate_distance(self, x1, y1, x2, y2):
        return int(round(math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))))


