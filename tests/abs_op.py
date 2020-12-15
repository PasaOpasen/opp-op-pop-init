import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import OppositionOperators, init_population, SampleInitializers
from plot_opposition import plot_opposition



min_bound = np.array([-4, -2])
max_bound = np.array([10, 16])


creator = SampleInitializers.Uniform(minimums = min_bound, maximums=max_bound)

points = init_population(total_count = 4, creator= creator)


oppositor = OppositionOperators.Continual.abs(minimums= min_bound, maximums= max_bound)

oppositions = OppositionOperators.Reflect(points, oppositor)



plot_opposition(points, oppositions, bounds = np.vstack((min_bound, max_bound)).T, title = r"$\bf{abs}$ oppositor operator", net = False, save_as = 'abs.png')

