# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 18:57:37 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from OppOpPopInit import init_population, SampleInitializers, set_seed
from OppOpPopInit.plotting import plot_pop

set_seed(1)

min_bound = np.array([-18, -11])
max_bound = np.array([16, 26])


creator = SampleInitializers.Combined(
    minimums=min_bound,
    maximums=max_bound,
    indexes= [
        [0],
        [1]
    ],
    creator_initializers=[
        SampleInitializers.Uniform,
        SampleInitializers.Normal
    ]
)


points = init_population(samples_count= 150, creator= creator)


plt.suptitle(r"$\bf{Uniform}$ for first dim and $\bf{Normal}$ for second dim")

plot_pop(points,  
                 bounds = np.vstack((min_bound-1, max_bound+1)).T,
                 title = "", net = False, 
                 save_as = './output/random_mixed_pop.png')