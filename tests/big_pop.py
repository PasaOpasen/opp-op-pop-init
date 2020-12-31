
import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import OppositionOperators, init_population, SampleInitializers



creator = SampleInitializers.Combined(
                minimums = np.zeros(1000),
                maximums= np.ones(1000),
                list_of_indexes = [np.array([]), np.arange(1000)],
                list_of_initializers_creators = [
                    SampleInitializers.RandomInteger,
                    SampleInitializers.Uniform
                ] )


pop = init_population(total_count = 100, creator = creator, oppositors = None) 