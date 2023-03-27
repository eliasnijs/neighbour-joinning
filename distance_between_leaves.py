import numpy as np


class DistanceMatrix:

    def __init__(self, data, **kwargs) -> None:
        N = len(data)
        if N < 2:
            raise ValueError('invalid distance matrix')
        for i in range(N):
            if data[i][i] != 0:
                raise ValueError('invalid distance matrix')
        self.data = np.array(data, **kwargs)

    def savetxt(self, filename: str, **kwargs) -> None:
        np.savetxt(filename, self.data, **kwargs)

    def limb_length(self, j):
        result = np.inf
        data = self.data
        for i, _ in enumerate(data):
            for k, _ in enumerate(data):
                if j in (i, k):
                    continue
                result = min((data[i][j] + data[j][k] - data[i][k]) / 2, result)
        return result

    def neighbour_joining(self):

        # NOTE(Elias):
        # d:    distance matrix
        # N:    length of the distance matrix
        # ids:  node id's of the rows/columns of d
        # e:    constructed edges
        d = self.data.copy()
        N = len(d)
        ids = np.array(range(N), dtype=np.int8)
        e = []

        for N_ in range(N, 2, -1):
            r = np.sum(d, axis=0)
            M = d.copy()
            for i in range(1, N_):
                for j in range(0, i):
                    M[i, j] -= (r[i] + r[j]) / (N_ - 2)

            # NOTE(Elias): s = (row, column) of the element in M with the lowest value
            s = np.unravel_index(np.argmin(M), M.shape)

            id_ = N + (N - N_)
            w1 = d[s] / 2 + (r[s[1]] - r[s[0]]) / (2 * (N_ - 2))
            w2 = d[s] - w1
            e.append((ids[s[0]], id_, w2))
            e.append((ids[s[1]], id_, w1))

            # NOTE(Elias): update data
            loss = d[s[0], s[1]]
            for i in range(N_):
                d[i, s[1]] = (d[s[1], i] + d[i, s[0]] - loss) / 2
                d[s[1], i] = (d[s[1], i] + d[i, s[0]] - loss) / 2
            d = np.delete(d, s[0], 0)
            d = np.delete(d, s[0], 1)

            # NOTE(Elias): update id's
            ids[s[1]] = id_
            ids = np.delete(ids, s[0], 0)

        # NOTE(Elias): connect the final 2 elements
        e.append((ids[0], ids[1], d[0][1]))

        return UnrootedTree(*e)

    def __str__(self) -> str:
        return str(self.data.tolist())

    def __repr__(self) -> str:
        return 'DistanceMatrix(' + str(self) + ')'

    @staticmethod
    def loadtxt(filename, **kwargs):
        data = np.loadtxt(filename, **kwargs)
        return DistanceMatrix(data)


class UnrootedTree:

    def __init__(self, *args):
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

    def path(self, pos, goal):
        result = self.__path(pos, goal, [pos], 0.0, {k: False for k in self.graph})
        return result[0] if result is not None else None

    def distance_matrix(self):
        leaves = dict(filter(lambda pair: len(pair[1]) == 1, self.graph.items()))
        data = np.zeros((len(leaves), len(leaves)))
        for i, n1 in enumerate(leaves):
            for j, n2 in enumerate(leaves):
                result = self.__path(n1, n2, [n1], 0.0, {k: False for k in self.graph})
                data[i][j] = result[1] if result is not None else 0.0
        return DistanceMatrix(data)

    def __str__(self):
        s = str(self.edges[0])
        for edge in self.edges[1:]:
            s += ', ' + str(edge)
        return s

    def __repr__(self):
        return 'UnrootedTree(' + str(self) + ')'

    @staticmethod
    def loadtxt(filename):
        edges = []
        with open(filename, 'r', encoding="utf-8") as file:
            line = file.readline()
            while line:
                line = line.strip().replace('<->', ':').split(':')
                edges.append((int(line[0]), int(line[1]), float(line[2])))
                line = file.readline()
        return UnrootedTree(*edges)
