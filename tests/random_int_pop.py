
import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import init_population, SampleInitializers
from plot_pop import plot_pop



min_bound = np.array([-8, -1])
max_bound = np.array([16, 26])


creator = SampleInitializers.RandomInteger(minimums = min_bound, maximums=max_bound)

points = init_population(total_count = 25, creator= creator)


plot_pop(points,  
                 bounds = np.vstack((min_bound, max_bound)).T, 
                 title = r"$\bf{RandomInteger}$ initializer", net = True, 
                 save_as = 'random_int_pop.png')