import numpy as np


class DistanceMatrix:

    def __init__(self, data, **kwargs) -> None:
        dim = len(data)
        if dim < 1:
            raise ValueError('invalid distance matrix')
        for i in range(dim):
            if data[i][i] != 0:
                raise ValueError('invalid distance matrix')
        self.data = np.array(data, dtype=np.int16, **kwargs)

    def savetxt(self, filename: str, **kwargs) -> None:
        np.savetxt(filename, self.data, **kwargs)

    def limb_length(self, j):
        result = np.inf
        data = self.data
        for i in range(len(data)):
            for k in range(len(data)):
                if i == j or k == j:
                    continue
                result = min((data[i][j] + data[j][k] - data[i][k]) / 2, result)
        return result

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
        with open(filename, 'r') as file:
            line = file.readline()
            while line:
                line = line.replace('<->', ',')
                line = line.replace(':', ',')
                edges += [eval(line)]
                line = file.readline()
        return UnrootedTree(*edges)
