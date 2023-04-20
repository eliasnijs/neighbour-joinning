import numpy as np


class DistanceMatrix:
    """
    a class DistanceMatrix that can be used to represent distance matrices. A distance matrix is a square symmetric
    matrix with diagonal elements equal to zero and positive off-diagonal elements.

    (C) Documentation: https://dodona.ugent.be/nl/courses/2278/series/24306/activities/582877927/
    """

    def __init__(self, data, **kwargs) -> None:
        """
        The initialisation method takes the same arguments as the numpy.array method. In case these arguments do not
        represent a valid distance matrix, a ValueError must be raised with the message invalid distance matrix.

        (C) Documentation: https://dodona.ugent.be/nl/courses/2278/series/24306/activities/582877927/
        """
        N = len(data)
        if N < 2:
            raise ValueError('invalid distance matrix')
        for i in range(N):
            if data[i][i] != 0:
                raise ValueError('invalid distance matrix')
        self.data = np.array(data, **kwargs)

    def savetxt(self, filename: str, **kwargs) -> None:
        """
        A method savetxt that can be used to write a distance matrix to a text file. The method takes a filename or file
        handle as a first argument and supports the same keyword arguments as the numpy.savetxt method. The distance
        matrix in the text file is formatted according to the specifications of the numpy.savetxt method.

        (C) Documentation: https://dodona.ugent.be/nl/courses/2278/series/24306/activities/582877927/
        """
        np.savetxt(filename, self.data, **kwargs)

    def limb_length(self, j) -> np.float64:
        """
        Method limb_length that takes the index of a row/column in the distance matrix and returns LimbLength(i).
        Given an additive matrix D and a leaf j:
        LimbLength(j) = (D_i,j + D_j,k - D_i,k) / 2 over all leaves i, k

        (C) Documentation: https://dodona.ugent.be/nl/courses/2278/series/24306/activities/814011908/
        """
        result = np.inf
        data = self.data
        for i, _ in enumerate(data):
            for k, _ in enumerate(data):
                if j in (i, k):
                    continue
                result = min((data[i][j] + data[j][k] - data[i][k]) / 2, result)
        return result

    def neighbour_joining(self):
        """
        Method that takes no arguments and returns the unrooted tree (an object of the class UnrootedTree) that results
        from applying the neighbour-joining algorithm to the distance matrix of the object.

        Example:
        Imagine the file 'distances.txt' with the following content:
        0   23  27  20
        23  0   30  28
        27  30  0   30
        20  28  30  0

        We can then do the following operations to get the unrooted tree:
        >>> D = DistanceMatrix.loadtxt('distances.txt')
        >>> D.neighbour_joining()
        UnrootedTree((0, 4, 8.0), (1, 5, 13.5), (2, 5, 16.5), (3, 4, 12.0), (4, 5, 2.0))

        (C) Documentation: https://dodona.ugent.be/nl/courses/2278/series/24306/activities/582877927/
        """
        # distance matrix
        d = self.data.copy()
        # length of the distance matrix
        N = len(d)
        # node id's of the rows/columns of d
        ids = np.array(range(N), dtype=np.int8)
        # constructed edges
        e = []

        for N_ in range(N, 2, -1):
            # figure out splitting point
            r = np.sum(d, axis=0)
            M = np.tile(r, (N_, 1))
            M = np.tril(d - (M + M.transpose()) / (N_ - 2), -1)

            # s = (row, column) of the element in M with the lowest value
            s = np.unravel_index(np.argmin(M), M.shape)
            print(s)

            # construct new edges
            id_ = N + (N - N_)
            w1 = d[s] / 2 + (r[s[1]] - r[s[0]]) / (2 * (N_ - 2))
            w2 = d[s] - w1
            e.append((ids[s[0]], id_, w2))
            e.append((ids[s[1]], id_, w1))

            # update data
            loss = d[s[0], s[1]]
            for i in range(N_):
                d[i, s[1]] = (d[s[1], i] + d[i, s[0]] - loss) / 2
                d[s[1], i] = (d[s[1], i] + d[i, s[0]] - loss) / 2
            d = np.delete(d, s[0], 0)
            d = np.delete(d, s[0], 1)

            ids[s[1]] = id_
            ids = np.delete(ids, s[0], 0)

        e.append((ids[0], ids[1], d[0][1]))
        return UnrootedTree(*e)

    @staticmethod
    def loadtxt(filename, **kwargs):
        """
        A static method loadtxt that can be used to read a distance matrix from a text file. The method takes the same
        arguments as the numpy.loadtxt method and must return an object of the class DistanceMatrix. The distance matrix
        in the text file must be formatted according to the specifications of the numpy.loadtxt method.

        (C) Documentation: https://dodona.ugent.be/nl/courses/2278/series/24306/activities/582877927/
        """
        data = np.loadtxt(filename, **kwargs)
        return DistanceMatrix(data)

    def __str__(self) -> str:
        return str(self.data.tolist())

    def __repr__(self) -> str:
        return 'DistanceMatrix(' + str(self) + ')'


class UnrootedTree:
    """
    A class UnrootedTree that can be used to represent unrooted trees whose nodes are uniquely labeled and whose edges
    are weighted with a floating point number.

    (C) Documentation: https://dodona.ugent.be/nl/courses/2278/series/24306/activities/582877927/
    """

    def __init__(self, *args):
        """
        The initialisation method takes zero or more edges. Each edge is represented as a tuple (i, j , w),
        with i and j the labels of the nodes connected by the edge and w the edge weight.

        (C) Documentation: https://dodona.ugent.be/nl/courses/2278/series/24306/activities/582877927/
        """
        graph = {}
        for edge in args:
            n1, n2, w = edge
            if n1 not in graph:
                graph[n1] = []
            if n2 not in graph:
                graph[n2] = []
            graph[n1].append((n2, float(w)))
            graph[n2].append((n1, float(w)))
        self.edges = [(n1, n2, float(w)) for n1, n2, w in args]
        self.graph = dict(sorted(graph.items()))

    def __path(self, pos, goal, path, w, marked):
        """
        A private helper method __path for constructing the path in method path(i, j);
        """
        if pos == goal:
            return path, w
        if marked[pos]:
            return None
        marked[pos] = True
        for n_next, w_next in self.graph[pos]:
            solution = self.__path(n_next, goal, path + [n_next], w + w_next, marked)
            if solution is not None:
                return solution
        return None

    def path(self, i, j):
        """
        A method path that takes two node labels i and j and returns a list containing the labels of the unique path
        between i and j in the unrooted tree.

        (C) Documentation: https://dodona.ugent.be/nl/courses/2278/series/24306/activities/582877927/
        """
        result = self.__path(i, j, [i], 0.0, {k: False for k in self.graph})
        return result[0] if result is not None else None

    def distance_matrix(self):
        """
        A method distance_matrix that returns a distance matrix (an object of the class DistanceMatrix) containing the
        pairwise distances between any pair of leaves in the unrooted tree. The rows and columns of the distance matrix
        correspond to the leaves of the unrooted tree, sorted in increasing order according to their unique label.

        (C) Documentation: https://dodona.ugent.be/nl/courses/2278/series/24306/activities/582877927/
        """
        leaves = dict(filter(lambda pair: len(pair[1]) == 1, self.graph.items()))
        data = np.zeros((len(leaves), len(leaves)))
        for i, n1 in enumerate(leaves):
            for j, n2 in enumerate(leaves):
                result = self.__path(n1, n2, [n1], 0.0, {k: False for k in self.graph})
                data[i][j] = result[1] if result is not None else 0.0
        return DistanceMatrix(data)

    @staticmethod
    def loadtxt(filename):
        """
        A static or class method loadtxt that can be used to read an unrooted tree from a text file. The method takes
        the file name or the file handle of the text file. The text file must contain a description of an unrooted tree
        as an adjacency list. Each line contains the description of an edge in the tree: the label , the symbols <->,
        the label , a colon (:) and weight . The method returns an object of the class UnrootedTree.

        (C) Documentation: https://dodona.ugent.be/nl/courses/2278/series/24306/activities/582877927/
        """
        edges = []
        with open(filename, 'r', encoding="utf-8") as file:
            line = file.readline()
            while line:
                line = line.strip().replace('<->', ':').split(':')
                edges.append((int(line[0]), int(line[1]), float(line[2])))
                line = file.readline()

        return UnrootedTree(*edges)

    def __str__(self):
        s = str(self.edges[0])
        for edge in self.edges[1:]:
            s += ', ' + str(edge)
        return s

    def __repr__(self):
        return 'UnrootedTree(' + str(self) + ')'