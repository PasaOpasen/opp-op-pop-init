# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 20:19:36 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import init_population, SampleInitializers, OppositionOperators, set_seed
from OppOpPopInit.plotting import plot_pop_op

set_seed(1)

min_bound = np.array([-38, -1])
max_bound = np.array([16, 100])

kwargs = dict(minimums = min_bound, maximums=max_bound)

for i in range(3):

    creator = SampleInitializers.Normal(**kwargs)

    oppositors = [
        OppositionOperators.Continual.over(**kwargs),
        OppositionOperators.Continual.quasi(**kwargs),
        OppositionOperators.Continual.abs(**kwargs)
        ]


    points = init_population(samples_count=20*(i+1), creator= creator, oppositors = oppositors)


    plot_pop_op(points,
                names = ['over reflection', 'quasi reflection', 'abs reflection'],
                     bounds = np.vstack((min_bound-1, max_bound+1)).T,
                     title = "Population with oppositions",
                net = False,
                     save_as = f'./output/pop_with_op{i+1}.png'
                )

