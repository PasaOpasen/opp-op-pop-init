# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 18:55:49 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import init_population, SampleInitializers
from plot_pop import plot_pop



min_bound = np.array([-8, -1])
max_bound = np.array([16, 26])


creator = SampleInitializers.Normal(minimums = min_bound, maximums=max_bound)

points = init_population(samples_count= 40, creator= creator)


plot_pop(points,  
                 bounds = np.vstack((min_bound, max_bound)).T, 
                 title = r"$\bf{Normal}$ initializer", net = False, 
                 save_as = 'random_normal_pop.png')