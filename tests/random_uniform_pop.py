# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 18:54:56 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import init_population, SampleInitializers, set_seed
from OppOpPopInit.plotting import plot_pop

set_seed(1)

min_bound = np.array([-8, -1])
max_bound = np.array([16, 26])


creator = SampleInitializers.Uniform(minimums = min_bound, maximums=max_bound)

points = init_population(samples_count= 50, creator= creator)


plot_pop(points,  
                 bounds = np.vstack((min_bound-1, max_bound+1)).T,
                 title = r"$\bf{Uniform}$ initializer", net = False, 
                 save_as = './output/random_uniform_pop.png')