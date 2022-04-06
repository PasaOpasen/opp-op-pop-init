
from typing import Callable, Optional, Sequence
import collections.abc

import numpy as np

from .initialiser import SampleInitializers
from .oppositor import OppositionOperators


def init_population(
    samples_count: int,
    creator: Callable[[], np.ndarray],
    oppositors: Optional[Sequence[Callable[[np.ndarray], np.ndarray]]] = None
) -> np.ndarray:
    """
    Returns population with size samples_count*dim
    using creator and oppositors for creator samples
    """

    assert int(samples_count) > 0, f"invalid samples_count argument: {samples_count}"

    if oppositors is None:
        return SampleInitializers.CreateSamples(creator, samples_count)
    assert isinstance(oppositors, collections.abc.Sequence)
    assert all(callable(opp) for opp in oppositors)

    groups = 1 + len(oppositors)

    if samples_count < groups:
        raise Exception(
            f"Not enough samples_count ({samples_count}) for this count of oppositors, needed {groups} at least"
        )

    group_size, tmp = divmod(samples_count, groups)
    init_pop = SampleInitializers.CreateSamples(creator, group_size + tmp)  # init samples
    samples_inds = np.arange(init_pop.shape[0])

    res = [init_pop] + [
        OppositionOperators.Reflect(init_pop[np.random.choice(samples_inds, group_size, replace=False), :], oppositor)
        for oppositor in oppositors
    ]

    return np.vstack(tuple(res))