# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:56:55 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import OppositionOperators, init_population, SampleInitializers
from plot_oppositions import plot_oppositions



min_bound = np.array([-8, -1])
max_bound = np.array([16, 26])


creator = SampleInitializers.Uniform(minimums = min_bound, maximums=max_bound)

point = init_population(total_count = 1, creator= creator)


oppositors = [
    OppositionOperators.Continual.over(minimums= min_bound, maximums= max_bound),
    OppositionOperators.Continual.quasi(minimums= min_bound, maximums= max_bound),
    OppositionOperators.Continual.quasi_reflect(minimums= min_bound, maximums= max_bound),
    OppositionOperators.Continual.abs(minimums= min_bound, maximums= max_bound),
    OppositionOperators.Continual.modular(minimums= min_bound, maximums= max_bound)
    ]


oppositions = np.array([opp(point[0,:]) for opp in oppositors])



plot_oppositions(point[0,:], 
                 oppositions, 
                 bounds = np.vstack((min_bound, max_bound)).T, 
                 names = ['over', 'quasi', 'quasi_reflect', 'abs', 'modular'],
                 title = "several oppositor operator", net = False, save_as = 'more_5.png')
