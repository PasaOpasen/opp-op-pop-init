
from typing import Sequence, Optional, Union, Callable

import collections.abc
import random
import numpy as np

from .utils import is_number, _check_mins_maxs


class SampleInitializers:

    @staticmethod
    def RandomInteger(
        minimums: Union[int, Sequence[int]],
        maximums: Union[int, Sequence[int]]
    ):
        """
        returns function creating random integer array between minimums and maximums 
        """
        mins, maxs = _check_mins_maxs(minimums, maximums, check_int=True)
        return lambda: np.array([random.randint(mn, mx) for mn, mx in zip(mins, maxs)])

    @staticmethod
    def Uniform(
        minimums: Union[float, Sequence[float]],
        maximums: Union[float, Sequence[float]]
    ):
        """
        returns function creating random array between minimums and maximums using uniform distribution 
        """
        mins, maxs = _check_mins_maxs(minimums, maximums)
        return lambda: np.array([random.uniform(mn, mx) for mn, mx in zip(mins, maxs)])

    @staticmethod
    def Normal(
        minimums: Union[float, Sequence[float]],
        maximums: Union[float, Sequence[float]],
        sd: Optional[Union[float, Sequence[float]]] = None
    ):
        """
        returns function creating random array between minimums and maximums using normal distribution 
        """

        mins, maxs = _check_mins_maxs(minimums, maximums)

        if sd is None:
            sd = np.array([abs(a - b)/6 for a, b in zip(mins, maxs)])
        elif is_number(sd):
            sd = np.full(mins.size, sd)
        elif isinstance(sd, collections.abc.Sequence):
            if len(sd) != mins.size:
                raise Exception(f"Invalid sizes of sd array ({len(sd)}) and minimums array ({mins.size})")
            sd = np.array(sd)
        else:
            raise Exception(f"Invalid parameter sd = {sd}. Should be None, number or sequence of numbers")
        
        centers = (mins + maxs) / 2

        def gauss(c, mn, mx, s):
            rd = np.random.normal(loc=c, scale=s)
            if rd < mn:
                return mn
            if rd > mx:
                return mx
            return rd
        
        return lambda: np.array([gauss(c, mn, mx, s) for c, mn, mx, s in zip(centers, mins, maxs, sd)])
    
    @staticmethod
    def Combined(
        minimums: Union[float, Sequence[float]],
        maximums: Union[float, Sequence[float]],
        indexes: Sequence[Sequence[int]],
        creator_initializers: Sequence[
            Callable[
                [Union[float, Sequence[float]], Union[float, Sequence[float]]],
                Callable[[], np.ndarray]
            ]
        ]
    ):
        """
        returns creator which creates vectors between minimums and maximums
        using pairs from creator_initializers and indexes to apply them
        """

        mins, maxs = _check_mins_maxs(minimums, maximums)

        inits_len = len(indexes)
        assert (inits_len == len(creator_initializers)), f"indexes list and initializers creators list must have equal length, gotten {inits_len} vs {len(creator_initializers)}"
        
        dim = mins.size
        all_indexes = set(list(range(dim)))

        dict_of_sets = {i: set(list(inds)) for i, inds in enumerate(indexes)}
        for i, s in dict_of_sets.items():
            if (s - all_indexes):
                raise Exception(
                    f"indexes should be between 0 and {dim-1} but {i}-element has {s - all_indexes}"
                )
        
        for i in range(inits_len-1):
            for j in range(i+1, inits_len):
                if dict_of_sets[i] & dict_of_sets[j]:
                    raise Exception(
                        f"there are common indexes between indexes from {i} element ({dict_of_sets[i]}) and indexes from {j} element ({dict_of_sets[j]})"
                    )
        
        # check union
        un = set()
        for s in dict_of_sets.values():
            un = un.union(s)
        if un != all_indexes:
            raise Exception(f"It's lack of indexes {all_indexes-un} in indexes argument")

        indexes_lst = [np.array(lst, dtype = np.int16) for lst in indexes]
        initializers = [
            initializer_creator(mins[ind], maxs[ind])
            for ind, initializer_creator in zip(indexes_lst, creator_initializers)
        ]

        def func():
            cp = np.empty(dim)
            for ind, creator in zip(indexes_lst, initializers):
                cp[ind] = creator()
            return cp

        return func
    
    @staticmethod
    def CreateSamples(creator: Callable[[], np.ndarray], count: int):
        """
        returns count objects (as 2D-array) using creator
        """
        return np.array([creator() for _ in range(int(count))])







