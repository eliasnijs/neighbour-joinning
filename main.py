import numpy as np
from distance_between_leaves import *


def main():
    # print(UnrootedTree.loadtxt('resources/edges_01.txt'))

    D = DistanceMatrix.loadtxt('resources/distances_02.txt')
    print(D.neighbour_joining())
    print(UnrootedTree((1, 5, 11.25), (0, 4, 8.75), (2, 4, 8.25), (4, 5, 0.25), (3, 5, 1.75)))


main()

# UnrootedTree((4, 0, 8.75), (4, 1, 11.25), (5, 2, 8.0),  (5, 3, 2.0),  (5, 4, 0.0))

