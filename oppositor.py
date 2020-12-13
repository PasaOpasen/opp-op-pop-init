import numpy as np 



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

    

    class Continual:
        pass





