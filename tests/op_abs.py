import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import OppositionOperators, init_population, SampleInitializers

from OppOpPopInit.plotting import plot_opposition
from OppOpPopInit import set_seed

set_seed(10)

min_bound = np.array([-4, -2])
max_bound = np.array([10, 16])


creator = SampleInitializers.Uniform(minimums=min_bound, maximums=max_bound)
points = init_population(samples_count=4, creator=creator)

oppositor = OppositionOperators.Continual.abs(minimums=min_bound, maximums=max_bound)
oppositions = OppositionOperators.Reflect(points, oppositor)

plot_opposition(points, oppositions, bounds=np.vstack((min_bound - 1, max_bound + 1)).T,
                title =r"$\bf{abs}$ oppositor operator", net=False, save_as='./output/abs.png')

