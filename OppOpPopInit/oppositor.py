
from typing import Sequence, Union, Tuple, Callable, Optional, List

import warnings

import random
import numpy as np 

from .aliases import TypeAlias, array1D, array2D, Number, IntBorder, FloatBorder, Border

from .utils import _check_mins_maxs


OppositorFunc: TypeAlias = Callable[[array1D], array1D]


class OppositionOperators:

    class Discrete:

        @staticmethod
        def _index_by_order(sizes: array1D) -> OppositorFunc:
            """
            for array of sizes [4, 5, 6]
            returns function which
                for indexes (from the end) [0, 3, 2]
                returns opposition indexes [4-1-0, 5-1-3, 6-1-2] = [3, 1, 3]
            """
            def func(indexes: array1D):
                return sizes - 1 - indexes
            return func
        
        @staticmethod
        def value_by_order(arrays: Sequence[array1D]) -> OppositorFunc:
            """
            for sequence of arrays [ [23, 26, 42] , [78, 53, 12, 11] ]
            returns function which
                for values [23, 12]
                returns opposition values [42, 53]
            it's like index_by_order but using values from array, not positions
            """
            arr_sizes = np.array([arr.size for arr in arrays])

            by_index = OppositionOperators.Discrete._index_by_order(arr_sizes)

            valid_lists: List[List[Number]] = [arr.tolist() for arr in arrays]

            def func(values: array1D) -> array1D:
                return by_index(
                    np.array([arr.index(val) for arr, val in zip(valid_lists, values)])
                )
            
            return func
        
        @staticmethod
        def integers_by_order(
            minimums: IntBorder,
            maximums: FloatBorder
        ) -> OppositorFunc:
            """
            returns like Continual abs 
            but for integer variables
            """
            mins, maxs = _check_mins_maxs(minimums, maximums, check_int=True)

            # shift to zero (get count of available cases by each dim)
            to_zero = maxs - mins + 1

            oppositor = OppositionOperators.Discrete._index_by_order(to_zero)
            return lambda array_of_values: mins + oppositor(array_of_values - mins)

    class Continual:

        @staticmethod
        def abs(
            minimums: FloatBorder,
            maximums: FloatBorder
        ) -> OppositorFunc:
            """
            absolute opposition
            for x between a and b returns (a+b-x)

            for zone [1, 4]x[2, 7]
            and point (2, 5)
            returns point (3, 4) 
            """

            mins, maxs = _check_mins_maxs(minimums, maximums)

            prep = mins + maxs

            def func(values: array1D):
                return prep - values
            
            return func
        
        @staticmethod
        def modular(
            minimums: FloatBorder,
            maximums: FloatBorder
        ) -> OppositorFunc:
            """
            modular opposition

            for x between a, b
            c = (a+b)/2
            returns a + (x + a - c) mod (b-a)
            """
            mins, maxs = _check_mins_maxs(minimums, maximums)
            
            diff = maxs - mins
            centers = (mins + maxs)/2

            bias = - centers + mins

            def func(values: array1D):
                return mins + (values + bias) % diff
            
            return func

        @staticmethod
        def quasi_reflect(
            minimums: FloatBorder,
            maximums: FloatBorder
        ) -> OppositorFunc:
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
            minimums: FloatBorder,
            maximums: FloatBorder
        ) -> OppositorFunc:
            """
            for x and c = (minimums + maximums)/2
            returns random uniform between abs_opposition(x) and c
            """

            mins, maxs = _check_mins_maxs(minimums, maximums)

            centers = (mins + maxs)/2
            
            abs_oppositor = OppositionOperators.Continual.abs(mins, maxs)

            def func(values: array1D):
                return np.array([random.uniform(c, x) for c, x in zip(centers, abs_oppositor(values))])
            
            return func     

        @staticmethod
        def over(
            minimums: FloatBorder,
            maximums: FloatBorder
        ) -> OppositorFunc:
            """
            for x and c = (minimums + maximums)/2
            returns random uniform between x and minimum if x > c and between x and maximum otherwise
            """

            mins, maxs = _check_mins_maxs(minimums, maximums)

            centers = (mins + maxs)/2

            def func(values: array1D):
                return np.array(
                    [
                        random.uniform(a, x) if x > c else random.uniform(b, x)
                        for a, b, c, x in zip(mins, maxs, centers, values)
                    ]
                )
            
            return func  

    @staticmethod
    def PartialOppositor(
        minimums: FloatBorder,
        maximums: FloatBorder,
        indexes_to_opp_creator: Sequence[
            Tuple[
                Sequence[int],
                Callable[
                    [array1D, array1D],
                    OppositorFunc
                ]
            ]
        ]
    ) -> OppositorFunc:
        """
        Partial oppositor for common minimums and maximums bounds

        creates small oppositor for some indexes of space for all oppositors
        and removes big oppositor which applies these small oppositors to their indexes

        indexes_to_opp_creator is a sequence of pairs like ([0, 1, 4], oppositor_creator)
        """
        mins, maxs = _check_mins_maxs(minimums, maximums)

        indexes_list = [np.array(t[0], dtype=np.int16) for t in indexes_to_opp_creator]
        creators_list = [t[1] for t in indexes_to_opp_creator]

        list_of_pairs = [
            (
                indexes,
                oppositor_creator(mins[indexes], maxs[indexes])
            )
            for indexes, oppositor_creator in zip(indexes_list, creators_list)
        ]

        return OppositionOperators.CombinedOppositor(list_of_pairs)

    @staticmethod
    def CombinedOppositor(
        indexes_to_oppositor: Sequence[
            Tuple[
                Sequence[int],
                OppositorFunc
            ]
        ]
    ) -> OppositorFunc:
        """
        Implementation of combined oppositor

        for sequence of pairs like [
            ([0, 1, 2], oppositor1) ,
            ([4, 6, 7], oppositor2) ,
            ([5, 8, 3], oppositor3)
        ]

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
                    # no exception cuz it should works with partial oppositor too
                    #raise Exception(f"indexes {inter} are common for pairs {i} and {j}")
                    warnings.warn(f"indexes {inter} are common for pairs {i} and {j}")

        def func(values: np.ndarray) -> np.ndarray:
            cp = values.copy()

            for indexes, oppositor in zip(arrays, oppositors):
                cp[indexes] = oppositor(cp[indexes])

            return cp

        return func


    @staticmethod
    def RandomPartialOppositor(
        list_of_count_step_oppositor_creator: Sequence[
            Tuple[
                int,
                int,
                Sequence[int],
                Callable[
                    [array1D, array1D],
                    OppositorFunc
                ]
            ]
        ],
        minimums: Border,
        maximums: Border
    ) -> OppositorFunc:
        """
        Returns random partial oppositor with these options:
        argument is the sequence of tuples like
        (how_many_indexes_for_current_oppositor, times_to_repeate_before_reinit, availavle indexes, oppositor)

        so it creates oppositor with random indexes for each small oppositor
        and repeates calculations some iteration before reinit random indexes for new oppositor
        """
        mins, maxs = _check_mins_maxs(minimums, maximums)

        # convert start data
        def _get_part(place: int):
            return [t[place] for t in list_of_count_step_oppositor_creator]

        random_counts = _get_part(0)
        steps = np.array(_get_part(1))
        available_indexes = _get_part(2)
        oppositors_creators = _get_part(3)

        current_counts = np.zeros(len(steps), dtype=np.int16)
        """counts of current usage by each part of oppositors"""

        assert all(count <= len(available) for count, available in zip(current_counts, available_indexes)), f"all available indexes must be with more len then count to sample them"
        available_indexes = [np.array(inds) for inds in available_indexes]
        """available indexes of each oppositor"""

        current_indexes = [
            np.random.choice(inds, count, replace=False)
            for inds, count in zip(available_indexes, random_counts)
        ]
        """indexes of oppositors"""
        oppositors = [
            op(mins[indexes], maxs[indexes])
            for op, indexes in zip(oppositors_creators, current_indexes)
        ]

        total_oppositor: OppositorFunc = None

        def recreate_oppositor():
            nonlocal total_oppositor
            total_oppositor = OppositionOperators.CombinedOppositor(
                [
                    (indexes, oppositor)
                    for indexes, oppositor in zip(current_indexes, oppositors)
                ]
            )
        recreate_oppositor()

        def func(values: array1D):
            nonlocal current_counts, current_indexes, oppositors

            answer = total_oppositor(values)
            current_counts += 1
            need_to_recreate: array1D = current_counts == steps

            # if it's needed to create some part of oppositor
            if need_to_recreate.sum() > 0:
                for i, need in enumerate(need_to_recreate):
                    if need:
                        # recount needed indexes
                        inds = np.random.choice(available_indexes[i], random_counts[i], replace=False)
                        current_indexes[i] = inds
                        current_counts[i] = 0  # reinit cuz it will be new oppositor
                        oppositors[i] = oppositors_creators[i](mins[inds], maxs[inds])

                recreate_oppositor()  # recreate our oppositor with some new parts

            return answer
        
        return func

    @staticmethod
    def Reflect(samples: array2D, oppositor: OppositorFunc) -> array2D:
        """
        for each sample in samples
        creates it's opposition using oppositor function

        samples is 2D numpy array with shape (samples, dimension) (samples by row)
        """
        return np.array(
            [oppositor(sample) for sample in samples]
        )

    @staticmethod
    def ReflectWithSelectionBest(
        population_samples: array2D,
        oppositor: OppositorFunc,
        eval_func: Callable[[array1D], float],
        samples_scores: Optional[array1D] = None,
        more_is_better: bool = False

    ) -> Tuple[array2D, array1D]:
        """
        Reflect N objects from population, evaluate scores and select best N from 2N

        Args:
            population_samples : 2D numpy array
                reflected population.
            oppositor: function
                applying oppositor.
            eval_func : function
                optimized function of population/task.
            samples_scores : None/1D numpy array, optional
                scores for reflected population (if calculated -- it's not necessary to calculate it again). The default is None.
            more_is_better : logical, optional
                The goal -- is maximize the function. The default is False.

        Returns:

            2d numpy array
                new population -- the best N from start N + reflected N objects.
            1d numpy array
                it's scores sorted from best to worse.
        """

        N = population_samples.shape[0]

        samples2 = OppositionOperators.Reflect(population_samples, oppositor)
        samples_total = np.vstack((population_samples, samples2))

        if samples_scores is None:
            scores = np.array([eval_func(samples_total[i]) for i in range(N*2)])
        else:
            scores = np.concatenate(
                (
                    samples_scores,
                    np.array([eval_func(samples2[i, :]) for i in range(N)])
                )
            )

        args = np.argsort(scores)
        args = args[-N::-1] if more_is_better else args[:N] 

        return samples_total[args, :], scores[args]






