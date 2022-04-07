import random
from typing import Union, Sequence, Tuple, Optional

import numpy as np


def is_number(x):

    if hasattr(x, '__len__'):
        return False

    try:
        return bool(0 == x*0)
    except:
        return False


def set_seed(seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def _check_mins_maxs(
    minimums: Union[int, float, Sequence[int], Sequence[float]],
    maximums: Union[int, float, Sequence[int], Sequence[float]],
    check_int: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    checks and fixes types, lengths, orders of minimums and maximums
    """
    min_is_number = is_number(minimums)
    max_is_number = is_number(maximums)

    if min_is_number and max_is_number:
        raise Exception("at least one of arguments must be a sequence of numbers")

    if not min_is_number and not max_is_number:
        assert len(minimums) == len(maximums), f"not equal length of minimums ({len(minimums)}) and maximums {len(maximums)}"

    result = np.minimum(minimums, maximums), np.maximum(minimums, maximums)
    if check_int:
        assert all(int(v) - v == 0 for v in result[0]+result[1]), "minimums and maximums must be int!"
        result = (
            result[0].astype(int),
            result[1].astype(int)
        )

    return result
