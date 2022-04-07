# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 22:19:49 2021

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import OppositionOperators, init_population, SampleInitializers, set_seed

set_seed(8)

dim = 10

minimums = np.full(dim, -5)
maximums = 5

creator = SampleInitializers.RandomInteger(minimums, maximums)

points = init_population(samples_count=10, creator=creator)

oppositor = OppositionOperators.RandomPartialOppositor([
     (1, 2, [0, 1, 2], OppositionOperators.Discrete.integers_by_order),
     (1, 1, [5, 6], OppositionOperators.Discrete.integers_by_order)
    ], 
    minimums, maximums
)

oppositions = OppositionOperators.Reflect(points, oppositor)

print(((oppositions - points) != 0)*1)

