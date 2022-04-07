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
from OppOpPopInit.plotting import plot_opposition
from OppOpPopInit import set_seed

set_seed(10)

min_bound = np.array([-8, -1])
max_bound = np.array([16, 26])


creator = SampleInitializers.Combined(
        minimums=min_bound, maximums=max_bound,
        indexes=([0], [1]),
        creator_initializers=[
            SampleInitializers.RandomInteger,
            SampleInitializers.Uniform
        ]
)

points = init_population(samples_count=4, creator=creator)

oppositor = OppositionOperators.CombinedOppositor(
    [
        ([0], OppositionOperators.Discrete.integers_by_order(
            minimums=[min_bound[0]],
            maximums=[max_bound[0]],
            )),
        ([1], OppositionOperators.Continual.over(
            minimums=[min_bound[1]],
            maximums=[max_bound[1]],
        ))
    ]
)

# in this situation it's equal to

# oppositor = OppositionOperators.PartialOppositor(
#     minimums=min_bound,
#     maximums=max_bound,
#     indexes_to_opp_creator=[
#         ([0], OppositionOperators.Discrete.integers_by_order),
#         ([1], OppositionOperators.Continual.over)
#     ]
# )


oppositions = OppositionOperators.Reflect(points, oppositor)

plt.suptitle(r"$\bf{over}$ for second dim and $\bf{integers\_by\_order}$ for first dim")
plot_opposition(points, oppositions,
                bounds=np.vstack((min_bound-1, max_bound+1)).T,
                title = "", net = True, save_as = './output/mixed.png')


