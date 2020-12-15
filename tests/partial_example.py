# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:33:43 2020

@author: qtckp
"""


import sys
sys.path.append('..')

import numpy as np
from OppOpPopInit import OppositionOperators




min_bound = np.array([-8, -3, -5.7, 0, 0])
max_bound = np.array([5, 4, 4, 9, 9])


points = np.array([
    [1, 2, 3, 4, 7.5],
    [1.6, -2, 3.9, 0.4, 5],
    [1.1, 3.2, -3, 4, 5],
    [4.1, 2, 3, -4, 0.5]
    ])

first_op_indexes = np.array([0, 2])
second_op_indexes = np.array([1, 3])

oppositor = OppositionOperators.PartialOppositor(
    [
        (first_op_indexes, OppositionOperators.Continual.abs(
            minimums= min_bound[first_op_indexes],
            maximums= max_bound[first_op_indexes],
            )),
        (second_op_indexes, OppositionOperators.Continual.over(
            minimums= min_bound[second_op_indexes],
            maximums= max_bound[second_op_indexes],
        ))
    ]
)

oppositions = OppositionOperators.Reflect(points, oppositor)

oppositions
