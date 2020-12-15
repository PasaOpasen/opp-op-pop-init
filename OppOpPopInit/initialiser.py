import numpy as np

from .helpers import isNumber
from .oppositor import OppositionOperators



class SampleInitializers:

    @staticmethod
    def RandomInteger(minimums, maximums):
        """
        returns function creating random integer array between minimums and maximums 
        """
        return lambda: np.array([np.random.randint(low = mn, high = mx) for mn, mx in zip(minimums, maximums)])

    
    @staticmethod
    def Uniform(minimums, maximums):
        """
        returns function creating random array between minimums and maximums using uniform distribution 
        """
        return lambda: np.array([np.random.uniform(low = mn, high = mx) for mn, mx in zip(minimums, maximums)])

    

    @staticmethod
    def Normal(minimums, maximums, sd = None):
        """
        returns function creating random array between minimums and maximums using normal distribution 
        """
        
        if sd is None:
            sd = np.array([abs(a - b)/6 for a, b in zip(minimums, maximums)])
        elif isNumber(sd):
            sd = np.full(minimums.size, sd)
        elif type(sd) == np.ndarray:
            if sd.size != minimums.size:
                raise Exception(f"Invalid sizes of sd array ({sd.size}) and minimums array ({minimums.size})")
        else:
            raise Exception(f"Invalid parameter sd = {sd}. Should be None, number or array of numbers")
        
        centers = (minimums + maximums) / 2

        def gauss(c, mn, mx, s):
            rd = np.random.normal(loc = c, scale = s)
            if rd < mn: return mn
            if rd > mx: return mx
            return rd
        
        return lambda: np.array([gauss(c, mn, mx, s) for c, mn, mx, s in zip(centers , minimums, maximums, sd)])
    
    @staticmethod
    def Combined(minimums, maximums, list_of_indexes, list_of_initializers_creators):
        """
        returns creator which creates vectors between minimums and maximums using list_of_initializers_creators creators for each list_of_indexes indexes of vector
        """

        assert (len(list_of_indexes) == len(list_of_initializers_creators)), "Indexes list and Initializers creators list must have equal length"
        
        dim = minimums.size

        all_indexes = set(list(range(dim)))

        dic_of_sets = {i: set(list(inds)) for i, inds in enumerate(list_of_indexes)}

        for i, s in dic_of_sets.items():
            if (s-all_indexes):
                raise Exception(f"indexes should be between 0 and {dim} but {i}-element has {s}")
        
        for i in range(dim-1):
            for j in range(i+1, dim):
                if dic_of_sets[i].intersection(dic_of_sets[j]):
                    raise Exception(f"there are common indexes between {i} element and indexes from {j} element")
        
        # check union
        un = set()
        for s in dic_of_sets.values():
            un = un.union(s)
        
        if un != all_indexes:
            raise Exception(f"It's lack of indexes {all_indexes-un} in list_of_indexes")


        indexes_lst = [np.array(lst) for lst in list_of_indexes]
        initializers = [initializer_creator(minimums[indexes], maximums[indexes]) for indexes, initializer_creator in zip(indexes_lst, list_of_initializers_creators)]


        def func():
            cp = np.empty(dim)

            for indexes, creator in zip(indexes_lst, initializers):
                cp[indexes] = creator()

            return cp

        return func
    
    @staticmethod
    def CreateSamples(creator, count):
        """
        returns count objects (as 2D-array) using creator
        """
        return np.array([creator() for _ in range(count)])




def init_population(total_count, creator, oppositors = None):
    """
    Returns population with size total_count*dim
    using creator and oppositors for creator samples
    """

    assert (type(total_count) == int and total_count > 0), f"Invalid total_count argument!"
    
    if oppositors is None:
        return SampleInitializers.CreateSamples(creator, total_count)
    
    groups = 1 + len(oppositors)

    if total_count < groups:
        raise Exception(f"Not enough total_count ({total_count}) for this count of oppositors, needed {groups} at least")
    
    group_size = total_count / groups
    tmp = total_count % groups

    init_pop = SampleInitializers.CreateSamples(creator, group_size + tmp)
    samples_inds = np.arange(init_pop.shape[0])

    res = [init_pop] + [OppositionOperators.Reflect(init_pop[np.random.choice(samples_inds, group_size, replace = False),:], oppositor) for oppositor in oppositors]

    return np.vstack(tuple(res))







