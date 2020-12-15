# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 18:57:37 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from OppOpPopInit import init_population, SampleInitializers
from plot_pop import plot_pop



min_bound = np.array([-18, -11])
max_bound = np.array([16, 26])


creator = SampleInitializers.Combined(
    minimums = min_bound, 
    maximums=max_bound,
    list_of_indexes= [np.array([0]), np.array([1])],
    list_of_initializers_creators=[
        SampleInitializers.Uniform,
        SampleInitializers.Normal
    ]
    )


points = init_population(total_count = 150, creator= creator)


plt.suptitle(r"$\bf{Uniform}$ for first dim and $\bf{Normal}$ for second dim")

plot_pop(points,  
                 bounds = np.vstack((min_bound, max_bound)).T, 
                 title = "", net = False, 
                 save_as = 'random_mixed_pop.png')