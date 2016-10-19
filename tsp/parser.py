import math

from vrp.cvrp import CVRP


class InstanceParser:

    def __init__(self, input_file):
        self.input_file = input_file

    def retrieve_instance(self):

        file_iter = self._read_file()
        while True:
            line = file_iter.next()
            if line == 'DIMENSION':
                file_iter.next()
                number_of_cities = int(file_iter.next())
            elif line == 'EDGE_WEIGHT_TYPE':
                file_iter.next()
                line = file_iter.next()
                if line != 'EUC_2D':
                    raise AttributeError('EDGE_WEIGHT_TYPE unsuported: %s' % line)
            elif line == 'NODE_COORD_SECTION':
                break

        self.x_coord = [None] * number_of_cities
        self.y_coord = [None] * number_of_cities

        for node in range(number_of_cities):
            node_id = int(file_iter.next())
            if node_id != node + 1:
                raise AttributeError('Uncontinous index of nodes in coordinates section')
            self.x_coord[node] = int(file_iter.next())
            self.y_coord[node] = int(file_iter.next())

        distance_matrix = self._calculate_distance_matrix(self.x_coord, self.y_coord)

        return distance_matrix

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


