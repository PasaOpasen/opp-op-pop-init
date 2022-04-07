# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:56:55 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import OppositionOperators, init_population, SampleInitializers
from OppOpPopInit.plotting import plot_oppositions

from OppOpPopInit import set_seed

set_seed(1)


min_bound = np.array([-8, -1])
max_bound = np.array([16, 26])

kwargs = dict(minimums = min_bound, maximums=max_bound)

creator = SampleInitializers.Uniform(**kwargs)
point = init_population(samples_count= 1, creator= creator)


oppositors = [
    OppositionOperators.Continual.over(**kwargs),
    OppositionOperators.Continual.quasi(**kwargs),
    OppositionOperators.Continual.quasi_reflect(**kwargs),
    OppositionOperators.Continual.abs(**kwargs),
    OppositionOperators.Continual.modular(**kwargs)
]

for i in range(5):

    oppositions = np.array([opp(point[0]) for opp in oppositors])

    plot_oppositions(
        point[0,:],
         oppositions,
         bounds = np.vstack((min_bound-1, max_bound+1)).T,
         names = ['over', 'quasi', 'quasi_reflect', 'abs', 'modular'],
         title = "several oppositor operator",
         net = False,
         save_as = f'./output/more_{i+1}.png'
    )
