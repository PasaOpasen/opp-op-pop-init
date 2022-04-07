import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import init_population, SampleInitializers, OppositionOperators, set_seed
from OppOpPopInit.plotting import plot_pop

set_seed(1)

def minimized_func(X):
    return np.sum(X[0::2]) - np.sum(X[1::2])

min_bound = np.array([-10, -10])
max_bound = np.array([16, 26])


creator = SampleInitializers.Uniform(minimums=min_bound, maximums=max_bound)

points = init_population(samples_count=25, creator=creator)

vals = np.array([minimized_func(points[i, :]) for i in range(points.shape[0])])

plot_pop(points,  
            bounds = np.vstack((min_bound-1, max_bound+1)).T,
                 title = f"Population at start\n mean score = {vals.mean()}", net = False, 
                 save_as = './output/reflection_with_selection_before.png'
         )
                 

better_points, better_vals = OppositionOperators.ReflectWithSelectionBest(
    points,
    oppositor = OppositionOperators.Continual.quasi(min_bound, max_bound),
    eval_func = minimized_func,
    samples_scores= vals
)


plot_pop(better_points,  
                 bounds = np.vstack((min_bound, max_bound)).T, 
                 title = f"Population after opposition and selection best N\n mean score = {better_vals.mean()}", net = False, 
                 save_as = './output/reflection_with_selection_after.png')




