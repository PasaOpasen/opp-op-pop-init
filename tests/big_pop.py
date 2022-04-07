
import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import init_population, SampleInitializers



creator = SampleInitializers.Combined(
                minimums=np.zeros(1000),
                maximums=1,
                indexes=[
                    np.array([999]),
                    np.arange(999)
                ],
                creator_initializers=[
                    SampleInitializers.RandomInteger,
                    SampleInitializers.Uniform
                ]
)


pop = init_population(samples_count=100, creator=creator, oppositors=None)

print(pop[:10, :10])

print(pop[:10, -10:])
