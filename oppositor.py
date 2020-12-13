import numpy as np 



class OppositionOperators:
    
    class Discrete:
        @staticmethod
        def index_by_arrange(array_of_sizes):
            """
            for array of sizes [4, 5, 6]
            and indexes [0, 3, 2]
            returns opposition indexes [4-1-0, 5-1-3, 6-1-2] = [3, 1, 3]
            """
            def func(array_of_indexes):
                return np.array([size - 1 - index for index, size in zip(array_of_indexes, array_of_sizes)])
            
            return func
    

    class Continual:
        pass





