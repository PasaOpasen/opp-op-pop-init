# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:37:37 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import OppositionOperators, init_population, SampleInitializers, set_seed
from OppOpPopInit.plotting import plot_opposition


set_seed(2)

min_bound = np.array([-8, -1])
max_bound = np.array([16, 26])


creator = SampleInitializers.Uniform(minimums = min_bound, maximums=max_bound)
points = init_population(samples_count=4, creator= creator)


oppositor = OppositionOperators.Continual.quasi(minimums=min_bound, maximums=max_bound)
oppositions = OppositionOperators.Reflect(points, oppositor)


plot_opposition(points, oppositions,
                bounds = np.vstack((min_bound-1, max_bound+1)).T, title = r"$\bf{quasi}$ oppositor operator", net = False,
                save_as = './output/quasi.png')
