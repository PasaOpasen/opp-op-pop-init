import random
from typing import Union, Sequence, Tuple, Optional, Any

import numpy as np

from .aliases import Border, array1D


def is_number(x: Any) -> bool:

    if hasattr(x, '__len__'):
        return False

    try:
        return bool(0 == x*0)

    except Exception:
        return False


def set_seed(seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def _check_mins_maxs(
    minimums: Border,
    maximums: Border,
    check_int: bool = False
) -> Tuple[array1D, array1D]:
    """
    checks and fixes types, lengths, orders of minimums and maximums

    Returns:
        tuple of minimums and maximums as arrays
    """
    min_is_number = is_number(minimums)
    max_is_number = is_number(maximums)

    if min_is_number and max_is_number:
        raise ValueError("at least one of arguments must be a sequence of numbers")

    if not min_is_number and not max_is_number:
        assert len(minimums) == len(maximums), f"not equal length of minimums ({len(minimums)}) and maximums {len(maximums)}"

    result = np.minimum(minimums, maximums), np.maximum(minimums, maximums)
    if check_int:
        assert all(int(v) - v == 0 for v in result[0].tolist() + result[1].tolist()), "minimums and maximums must be int!"
        result = (
            result[0].astype(int),
            result[1].astype(int)
        )

    return result
