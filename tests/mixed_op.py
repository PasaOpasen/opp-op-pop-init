# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:50:46 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from OppOpPopInit import OppositionOperators, init_population, SampleInitializers
from plot_opposition import plot_opposition



min_bound = np.array([-8, -1])
max_bound = np.array([16, 26])


creator = SampleInitializers.Combined(
        minimums = min_bound, maximums=max_bound,
        list_of_indexes= ([0], [1]),
        list_of_initializers_creators= [
            SampleInitializers.RandomInteger,
            SampleInitializers.Uniform
        ]
        )

points = init_population(total_count = 4, creator= creator)

oppositor = OppositionOperators.PartialOppositor(
    [
        ([0], OppositionOperators.Discrete.integers_by_order(
            minimums= np.array([min_bound[0]]),
            maximums= np.array([max_bound[0]]),
            )),
        ([1], OppositionOperators.Continual.over(
            minimums= np.array([min_bound[1]]),
            maximums= np.array([max_bound[1]]),
        ))
    ]
)

oppositions = OppositionOperators.Reflect(points, oppositor)


plt.suptitle(r"$\bf{over}$ for second dim and $\bf{integers\_by\_order}$ for first dim")
plot_opposition(points, oppositions, bounds = np.vstack((min_bound, max_bound)).T, title = "", net = True, save_as = 'mixed.png')


