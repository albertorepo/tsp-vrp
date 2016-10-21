import math

class InstanceParser:
    """Parser of TSP files.

    Read the *.tsp files and return the main values asociated
    with the instance of problem defined in the file.
    If the file is not well.built, it will raise a ValueError.

    Parameters
    ----------
    input_file : string
        Path to the tsp file that contains the information
        about the instance of the TSP.

    Attributes
    ----------
    x_coord_ : array-like, shape = (n_cities)
        Coordinate X of each city of the problem.

    y_coord_ : array-like, shape = (n_cities)
        Coordinate Y of each city of the problem.
    """

    def __init__(self,
                 input_file):
        self.input_file = input_file
        self.x_coord_ = None
        self.y_coord_ = None

    def retrieve_instance(self):
        """Read the file and retrieve the information
        related to the instance.

        Parameters
        ----------

        Returns
        -------
        distance_matrix : array-like, shape = (n_cities, n_cities)
            Distances between each of the cities of the problem.
        """

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
                    raise ValueError('EDGE_WEIGHT_TYPE unsuported: %s' % line)
            elif line == 'NODE_COORD_SECTION':
                break

        self.x_coord_ = [None] * number_of_cities
        self.y_coord_ = [None] * number_of_cities

        for node in range(number_of_cities):
            node_id = int(file_iter.next())
            if node_id != node + 1:
                raise ValueError('Uncontinous index of nodes in coordinates section')
            self.x_coord_[node] = int(file_iter.next())
            self.y_coord_[node] = int(file_iter.next())

        distance_matrix = self._calculate_distance_matrix()

        return distance_matrix

    def _read_file(self):
        """Read the file and returns the content

        Returns
        -------
        tokens : iterator
            Returns an iterator of each word in the file.
        """
        with open(self.input_file) as f:
            for token in f.read().split():
                yield token

    def _calculate_distance_matrix(self):
        """Calculate the distances between the cities.

        Returns
        -------
        distance_matrix : array-like, shape = (n_cities, n_cities)
            Distances between each of the cities of the problem.
        """
        number_of_nodes = len(self.x_coord_)
        distance_matrix = [[None for i in xrange(number_of_nodes)] for y in xrange(number_of_nodes)]

        for i in range(number_of_nodes):
            for j in range(number_of_nodes):
                distance = self._calculate_distance(self.x_coord_[i], self.y_coord_[i],
                                                    self.x_coord_[j], self.y_coord_[j])
                distance_matrix[i][j] = distance
        return distance_matrix

    def _calculate_distance(self, x1, y1, x2, y2):
        """Calculate the euclidean distance between two points."""
        return int(round(math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))))


