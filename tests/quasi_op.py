# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:37:37 2020

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

points = init_population(total_count = 4, creator= creator)


oppositor = OppositionOperators.Continual.quasi(minimums= min_bound, maximums= max_bound)

oppositions = OppositionOperators.Reflect(points, oppositor)



plot_opposition(points, oppositions, bounds = np.vstack((min_bound, max_bound)).T, title = r"$\bf{quasi}$ oppositor operator", net = False, save_as = 'quasi.png')
