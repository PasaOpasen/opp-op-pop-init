import numpy as np 
from .helpers import isNumber

class OppositionOperators:

    class Discrete:
        @staticmethod
        def index_by_order(array_of_sizes):
            """
            for array of sizes [4, 5, 6]
            and indexes [0, 3, 2]
            returns opposition indexes [4-1-0, 5-1-3, 6-1-2] = [3, 1, 3]
            """
            def func(array_of_indexes):
                return np.array([size - 1 - index for index, size in zip(array_of_indexes, array_of_sizes)])
            
            return func
        
        @staticmethod
        def value_by_order(list_of_valid_arrays):
            """
            for list of arrays [ [23, 26, 42] , [78, 53, 12, 11] ]
            and values [23, 12]
            returns opposition values [42, 53] 
            it's like index_by_order but using values from array, not positions
            """
            arr_sizes = np.array([len(arr) for arr in list_of_valid_arrays])

            by_index = OppositionOperators.Discrete.index_by_order(arr_sizes)

            list_of_valid_lists = [list(arr) for arr in list_of_valid_arrays]

            def func(array_of_values):
                return by_index(np.array([arr.index(val) for arr, val in zip(list_of_valid_lists, array_of_values)]))
            
            return func
        
        @staticmethod
        def integers_by_order(minimums, maximums):
            """
            returns like Continual abs 
            but for integer variables
            """
            
            assert (type(minimums) == int or type(maximums) == int or minimums.size == maximums.size), f"Invalid sizes of bounds! {minimums.size} for first and {maximums.size} for second"

            to_zero = (maximums - minimums)

            oppositor = OppositionOperators.Discrete.index_by_order(to_zero)

            return lambda array_of_values: minimums + oppositor(array_of_values - minimums)





    

    class Continual:
        @staticmethod
        def __assert_sizes(min_arr, max_arr):
            assert (isNumber(min_arr) or isNumber(max_arr) or min_arr.size == max_arr.size), f"Invalid sizes of bounds! {min_arr.size} for first and {max_arr.size} for second"

        @staticmethod
        def abs(minimums, maximums):
            """
            absolute opposition
            for x between a and b returns (a+b-x)

            for zone [1, 4]x[2, 7]
            and point (2, 5)
            returns point (3, 4) 
            """

            OppositionOperators.Continual.__assert_sizes(minimums, maximums)

            prep = minimums + maximums
            def func(array_of_values):
                return prep - array_of_values
            
            return func
        
        @staticmethod
        def modular(minimums, maximums):
            """
            modular opposition

            for x between a, b
            c = (a+b)/2
            returns (x + a - c) mod (b-a)
            """
            OppositionOperators.Continual.__assert_sizes(minimums, maximums)
            
            diff = maximums - minimums
            centers = (minimums + maximums)/2

            bias = centers - minimums

            def func(array_of_values):
                return (array_of_values + bias) % diff
            
            return func

        @staticmethod
        def quasi_reflect(minimums, maximums):
            """
            for x and c = (minimums + maximums)/2
            returns random uniform between x and c
            """

            OppositionOperators.Continual.__assert_sizes(minimums, maximums)

            centers = (minimums + maximums)/2

            def func(array_of_values):
                return np.array([np.random.uniform(c, x) for c, x in zip(centers, array_of_values)])
            
            return func
        
        @staticmethod
        def quasi(minimums, maximums):
            """
            for x and c = (minimums + maximums)/2
            returns random uniform between abs_opposition(x) and c
            """

            OppositionOperators.Continual.__assert_sizes(minimums, maximums)

            centers = (minimums + maximums)/2
            
            abs_oppositor = OppositionOperators.Continual.abs(minimums, maximums)

            def func(array_of_values):
                return np.array([np.random.uniform(c, x) for c, x in zip(centers, abs_oppositor(array_of_values))])
            
            return func     

        @staticmethod
        def over(minimums, maximums):
            """
            for x and c = (minimums + maximums)/2
            returns random uniform between x and minimum if x > c and between x and maximum otherwise
            """

            OppositionOperators.Continual.__assert_sizes(minimums, maximums)

            centers = (minimums + maximums)/2

            def func(array_of_values):
                return np.array([np.random.uniform(a, x) if x > c else np.random.uniform(b, x) for a, b, c, x in zip(minimums, maximums, centers, array_of_values)])
            
            return func  


        @staticmethod
        def Partial(minimums, maximums, list_of_pairs_inds_vs_oppositor_creators):
            """
            Partial oppositor for continual space and common minimums and maximums bounds

            list_of_pairs is list of pairs like ([0, 1, 4], oppositor_creator)
            """
            OppositionOperators.Continual.__assert_sizes(minimums, maximums)

            if isNumber(minimums):
                minimums = np.full(maximums.size, minimums)
            elif isNumber(maximums):
                maximums = np.full(minimums.size, maximums)

            indexes_list = [np.array(t[0]) for t in list_of_pairs_inds_vs_oppositor_creators]
            creators_list = [t[1] for t in list_of_pairs_inds_vs_oppositor_creators]

            list_of_pairs = [(indexes, oppositor_creator(minimums[indexes], maximums[indexes])) for indexes, oppositor_creator in zip(indexes_list, creators_list)]

            return OppositionOperators.PartialOppositor(list_of_pairs)



    @staticmethod
    def PartialOppositor(list_of_pairs_inds_vs_oppositor):
        """
        Implementation of partial oppositor

        for list/tuple of pairs like [([0, 1, 2], oppositor1) ,  ([4, 6, 7], oppositor2) , ([5, 8, 3], oppositor3) ,]

        returns partial oppositor applying these oppositors for these indexes
        """

        arrays = [np.array(t[0]) for t in list_of_pairs_inds_vs_oppositor]
        oppositors = [t[1] for t in list_of_pairs_inds_vs_oppositor]

        dic_of_sets = { }
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
                
                inter = dic_of_sets[i].intersection(dic_of_sets[j])

                if inter:
                    raise Exception(f"indexes {inter} are common for pairs {i} and {j}")
        

        def func(array_of_values):
            cp = array_of_values.copy()

            for indexes, oppositor in zip(arrays, oppositors):
                cp[indexes] = oppositor(cp[indexes])

            return cp

        return func
    
    @staticmethod
    def Reflect(samples, oppositor):
        """
        for each sample in samples creates it's opposition using oppositor function

        samples is 2D numpy array with shape (samples, dimension)
        """
        return np.array([oppositor(samples[i, :]) for i in range(samples.shape[0])])








