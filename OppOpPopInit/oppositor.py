
from typing import Sequence, Union, Tuple, Callable

import warnings

import random
import numpy as np 

from .utils import is_number, _check_mins_maxs


class OppositionOperators:

    class Discrete:

        @staticmethod
        def _index_by_order(sizes: np.ndarray):
            """
            for array of sizes [4, 5, 6]
            returns function which
                for indexes (from the end) [0, 3, 2]
                returns opposition indexes [4-1-0, 5-1-3, 6-1-2] = [3, 1, 3]
            """
            def func(indexes: np.ndarray):
                return sizes - 1 - indexes
            return func
        
        @staticmethod
        def value_by_order(arrays: Sequence[np.ndarray]):
            """
            for list of arrays [ [23, 26, 42] , [78, 53, 12, 11] ]
            returns function which
                for values [23, 12]
                returns opposition values [42, 53]
            it's like index_by_order but using values from array, not positions
            """
            arr_sizes = np.array([arr.size for arr in arrays])

            by_index = OppositionOperators.Discrete._index_by_order(arr_sizes)

            valid_lists = [arr.tolist() for arr in arrays]

            def func(values: np.ndarray):
                return by_index(
                    np.array([arr.index(val) for arr, val in zip(valid_lists, values)])
                )
            
            return func
        
        @staticmethod
        def integers_by_order(
            minimums: Union[int, Sequence[int]],
            maximums: Union[int, Sequence[int]]
        ):
            """
            returns like Continual abs 
            but for integer variables
            """
            mins, maxs = _check_mins_maxs(minimums, maximums, check_int=True)

            to_zero = maxs - mins

            oppositor = OppositionOperators.Discrete._index_by_order(to_zero)
            return lambda array_of_values: mins + oppositor(array_of_values - mins)

    class Continual:

        @staticmethod
        def abs(
            minimums: Union[float, Sequence[float]],
            maximums: Union[float, Sequence[float]]
        ):
            """
            absolute opposition
            for x between a and b returns (a+b-x)

            for zone [1, 4]x[2, 7]
            and point (2, 5)
            returns point (3, 4) 
            """

            mins, maxs = _check_mins_maxs(minimums, maximums)

            prep = mins + maxs

            def func(values: np.ndarray):
                return prep - values
            
            return func
        
        @staticmethod
        def modular(
            minimums: Union[float, Sequence[float]],
            maximums: Union[float, Sequence[float]]
        ):
            """
            modular opposition

            for x between a, b
            c = (a+b)/2
            returns (x + a - c) mod (b-a)
            """
            mins, maxs = _check_mins_maxs(minimums, maximums)
            
            diff = maxs - mins
            centers = (mins + maxs)/2

            bias = centers - mins

            def func(values: np.ndarray):
                return (values + bias) % diff
            
            return func

        @staticmethod
        def quasi_reflect(
            minimums: Union[float, Sequence[float]],
            maximums: Union[float, Sequence[float]]
        ):
            """
            for x and c = (minimums + maximums)/2
            returns random uniform between x and c
            """

            mins, maxs = _check_mins_maxs(minimums, maximums)

            centers = (mins + maxs)/2

            def func(values: np.ndarray):
                return np.array([random.uniform(c, x) for c, x in zip(centers, values)])
            
            return func
        
        @staticmethod
        def quasi(
            minimums: Union[float, Sequence[float]],
            maximums: Union[float, Sequence[float]]
        ):
            """
            for x and c = (minimums + maximums)/2
            returns random uniform between abs_opposition(x) and c
            """

            mins, maxs = _check_mins_maxs(minimums, maximums)

            centers = (mins + maxs)/2
            
            abs_oppositor = OppositionOperators.Continual.abs(mins, maxs)

            def func(values: np.ndarray):
                return np.array([random.uniform(c, x) for c, x in zip(centers, abs_oppositor(values))])
            
            return func     

        @staticmethod
        def over(
            minimums: Union[float, Sequence[float]],
            maximums: Union[float, Sequence[float]]
        ):
            """
            for x and c = (minimums + maximums)/2
            returns random uniform between x and minimum if x > c and between x and maximum otherwise
            """

            mins, maxs = _check_mins_maxs(minimums, maximums)

            centers = (mins + maxs)/2

            def func(values: np.ndarray):
                return np.array([random.uniform(a, x) if x > c else random.uniform(b, x) for a, b, c, x in zip(mins, maxs, centers, values)])
            
            return func  

        @staticmethod
        def Partial(
            minimums: Union[float, Sequence[float]],
            maximums: Union[float, Sequence[float]],
            indexes_to_opp_creator: Sequence[
                Tuple[
                    Sequence[int],
                    Callable[
                        [np.ndarray, np.ndarray],
                        Callable[[np.ndarray], np.ndarray]
                    ]
                ]
            ]
        ):
            """
            Partial oppositor for continual space and common minimums and maximums bounds

            indexes_to_opp_creator is list of pairs like ([0, 1, 4], oppositor_creator)
            """
            mins, maxs = _check_mins_maxs(minimums, maximums)

            indexes_list = [np.array(t[0], dtype=np.int16) for t in indexes_to_opp_creator]
            creators_list = [t[1] for t in indexes_list]

            list_of_pairs = [
                (indexes, oppositor_creator(mins[indexes], maxs[indexes]))
                for indexes, oppositor_creator in zip(indexes_list, creators_list)
            ]

            return OppositionOperators.PartialOppositor(list_of_pairs)

    @staticmethod
    def PartialOppositor(
        indexes_to_oppositor: Sequence[
            Tuple[
                Sequence[int],
                Callable[[np.ndarray], np.ndarray]
            ]
        ]
    ):
        """
        Implementation of partial oppositor

        for sequence of pairs like [([0, 1, 2], oppositor1) ,  ([4, 6, 7], oppositor2) , ([5, 8, 3], oppositor3) ,]

        returns partial oppositor applying these oppositors for these indexes
        """

        arrays = [np.array(t[0], dtype=np.int16) for t in indexes_to_oppositor]
        oppositors = [t[1] for t in indexes_to_oppositor]

        dic_of_sets = {}
        # check repeats
        for i, arr in enumerate(arrays):
            lst = list(arr)
            st = set(lst)
            if len(lst) > len(st):
                raise Exception(f"there are repeated indexes at {i} pair")
            dic_of_sets[i] = st
        
        # check intersections
        for i in range(len(arrays) - 1):
            for j in range(i+1, len(arrays)):
                inter = dic_of_sets[i] & dic_of_sets[j]
                if inter:
                    #raise Exception(f"indexes {inter} are common for pairs {i} and {j}")
                    warnings.warn(f"indexes {inter} are common for pairs {i} and {j}")

        def func(values: np.ndarray) -> np.ndarray:
            cp = values.copy()

            for indexes, oppositor in zip(arrays, oppositors):
                cp[indexes] = oppositor(cp[indexes])

            return cp

        return func

    ########################################################################

    @staticmethod
    def RandomPartialOppositor(list_of_count_step_oppositor_creator, minimums, maximums, total_dim):
        """
        Returns random partial oppositor with these options:
        argument is the list of tuples like (how_many_indexes_for_current_oppositor, times_to_repeate_before_reinit, oppositor)
        so it creates oppositor with random indexes for each oppositor and repeates calculations some iteration before reinit random indexes for new oppositor
        """

        # convert start data
        def get_part(place):
            return [t[place] for t in list_of_count_step_oppositor_creator]

        random_counts = get_part(0)
        steps = np.array(get_part(1))
        oppositors_creators = get_part(2)

        # local variables
        all_indexes = np.arange(total_dim)
        need_to_recreate = np.zeros(len(steps), dtype = np.bool) # flag of need to recreate some part of partial oppositor
        current_counts = np.zeros(len(steps), dtype = np.int16) # counts of current usage by each part of oppositors
        current_indexes = [np.random.choice(all_indexes, count, replace = False) for count in random_counts] # indexes of oppositors
        oppositors = [op(minimums[indexes], maximums[indexes]) for op, indexes in zip(oppositors_creators, current_indexes)]

        total_oppositor = lambda tmp:None
        def recreate_oppositor():
            nonlocal total_oppositor
            total_oppositor = OppositionOperators.PartialOppositor([[indexes, oppositor] for indexes, oppositor in zip(current_indexes, oppositors)])
        
        recreate_oppositor()

        def func(array_of_values):
            nonlocal current_counts, current_indexes, need_to_recreate, oppositors

            answer = total_oppositor(array_of_values)
            #raise Exception()
            current_counts += 1
            need_to_recreate = current_counts == steps

            # if it's needed to create some part of oppositor
            if np.sum(need_to_recreate) > 0:
                for i, need in enumerate(need_to_recreate):
                    if need:
                        current_indexes[i] = np.random.choice(all_indexes, random_counts[i], replace = False) # recount needed indexes
                        current_counts[i] = 0 # reinit cuz it will be new oppositor
                        oppositors[i] = oppositors_creators[i](minimums[current_indexes[i]], maximums[current_indexes[i]])
                
                # reinit some data
                need_to_recreate = np.zeros_like(need_to_recreate, dtype = np.bool)

                recreate_oppositor() # recreate our oppositor with some new parts

            return answer
        
        return func

    @staticmethod
    def Reflect(samples, oppositor):
        """
        for each sample in samples creates it's opposition using oppositor function

        samples is 2D numpy array with shape (samples, dimension)
        """
        return np.array([oppositor(samples[i, :]) for i in range(samples.shape[0])])

    @staticmethod
    def ReflectWithSelectionBest(population_samples, oppositor, eval_func, samples_scores = None, more_is_better = False):
        """
        Reflect N objects from population, evaluate scores and select best N from 2N

        Parameters
        ----------
        population_samples : 2D numpy array
            reflected population.
        oppositor : function
            applying oppositor.
        eval_func : function
            optimized function of population/task.
        samples_scores : None/1D numpy array, optional
            scores for reflected population (if calculated -- it's not necessary to calculate it again). The default is None.
        more_is_better : logical, optional
            The goal -- is maximize the function. The default is False.

        Returns
        -------
        2d numpy array
            new population -- the best N from start N + reflected N objects.
        1d numpy array
            it's scores sorted from best to worse.
        """

        N = population_samples.shape[0]

        samples2 = OppositionOperators.Reflect(population_samples, oppositor)

        samples_total = np.vstack((population_samples, samples2))

        if samples_scores is None:
            scores = np.array([eval_func(samples_total[i, :]) for i in range(N*2)])
        else:
            scores = np.concatenate((samples_scores, np.array([eval_func(samples2[i, :]) for i in range(N)])))

        args = np.argsort(scores)
        args = args[-N::-1] if more_is_better else args[:N] 

        return samples_total[args, :], scores[args]






