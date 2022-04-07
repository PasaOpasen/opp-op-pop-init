# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:44:40 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import OppositionOperators, init_population, SampleInitializers

from OppOpPopInit.plotting import plot_opposition
from OppOpPopInit import set_seed

set_seed(1)


min_bound = np.array([-8, -1])
max_bound = np.array([16, 26])


creator = SampleInitializers.RandomInteger(minimums=min_bound, maximums=max_bound)
points = init_population(samples_count=3, creator=creator)


oppositor = OppositionOperators.Discrete.integers_by_order(minimums=min_bound, maximums=max_bound)
oppositions = OppositionOperators.Reflect(points, oppositor)


plot_opposition(points, oppositions,
                bounds=np.vstack((min_bound - 1, max_bound + 1)).T,
                title=r"$\bf{integers\_by\_order}$ oppositor operator",
                net=True, save_as='./output/integers_by_order.png')
