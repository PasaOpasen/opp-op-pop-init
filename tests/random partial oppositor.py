# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 22:19:49 2021

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from OppOpPopInit import OppositionOperators, init_population, SampleInitializers
from plot_opposition import plot_opposition


dim = 10

minimums = np.full(dim, -5)
maximums = np.full(dim, 5)

creator = SampleInitializers.RandomInteger(minimums, maximums)

points = init_population(total_count = 6, creator= creator)

oppositor = OppositionOperators.RandomPartialOppositor([
     (1, 2, OppositionOperators.Discrete.integers_by_order),
     (1, 1, OppositionOperators.Discrete.integers_by_order)
    ], 
    minimums, maximums,
    total_dim = dim)

oppositions = OppositionOperators.Reflect(points, oppositor)

print(((oppositions - points) != 0)*1)

