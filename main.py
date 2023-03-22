import numpy as np
from distance_between_leaves import *




D = DistanceMatrix.loadtxt('resources/distances_01.txt')
print(D.limb_length(0), 11.0)
