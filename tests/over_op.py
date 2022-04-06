# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:40:34 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import OppositionOperators, init_population, SampleInitializers
from plot_opposition import plot_opposition



min_bound = np.array([-8, -1])
max_bound = np.array([16, 26])


creator = SampleInitializers.Uniform(minimums = min_bound, maximums=max_bound)

points = init_population(samples_count= 5, creator= creator)


oppositor = OppositionOperators.Continual.over(minimums= min_bound, maximums= max_bound)

oppositions = OppositionOperators.Reflect(points, oppositor)



plot_opposition(points, oppositions, bounds = np.vstack((min_bound, max_bound)).T, title = r"$\bf{over}$ oppositor operator", net = False, save_as = 'over.png')
