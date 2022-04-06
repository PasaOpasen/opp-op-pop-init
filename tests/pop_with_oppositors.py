# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 20:19:36 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import init_population, SampleInitializers, OppositionOperators
from plot_pop_op import plot_pop_op



min_bound = np.array([-38, -1])
max_bound = np.array([16, 100])


creator = SampleInitializers.Normal(minimums = min_bound, maximums=max_bound)

oppositors = [
    OppositionOperators.Continual.over(minimums= min_bound, maximums= max_bound),
    OppositionOperators.Continual.quasi(minimums= min_bound, maximums= max_bound),
    OppositionOperators.Continual.abs(minimums= min_bound, maximums= max_bound)
    ]


points = init_population(samples_count= 60, creator= creator, oppositors = oppositors)


plot_pop_op(points, 
            names = ['over reflection', 'quasi reflection', 'abs reflection'],
                 bounds = np.vstack((min_bound, max_bound)).T, 
                 title = "Population with oppositions", net = False, 
                 save_as = 'pop_with_op3.png')

