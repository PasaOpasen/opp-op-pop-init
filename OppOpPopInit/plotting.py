from typing import Tuple, Optional, Sequence, List

import numpy as np


def _set_axis(
    bounds: Tuple[
        Tuple[float, float],
        Tuple[float, float]
    ]
):
    import matplotlib.pyplot as plt

    xmin, xmax, ymin, ymax = bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]
    result = (xmin, xmax, ymin, ymax)
    plt.axis(result)

    return result


def _plot_net(xmin: float, xmax: float, ymin: float,  ymax: float):

    import matplotlib.pyplot as plt

    xs = np.arange(round(xmin), round(xmax) + 1)
    ys = np.arange(round(ymin), round(ymax) + 1)

    for xc in xs:
        plt.axvline(x=xc, color='k', linestyle='--', lw=0.3)

    for yc in ys:
        plt.axhline(y=yc, color='k', linestyle='--', lw=0.3)


def _save_fig(save_as: Optional[str]):
    if save_as is not None:
        import matplotlib.pyplot as plt
        plt.savefig(save_as, dpi=200)


def _random_coords(xmin: float, xmax: float, ymin: float, ymax: float, wd: float = 0.2):
    rx, ry = tuple(np.random.uniform(0.5 - wd, 0.5 + wd, 2))
    return (
        xmin * rx + (1 - rx) * xmax,
        ymin * ry + (1 - ry) * ymax
    )


def plot_opposition(
    points: np.ndarray,
    oppositions: np.ndarray,
    bounds: Tuple[
        Tuple[float, float],
        Tuple[float, float]
    ],
    title: str,
    net: bool = False,
    save_as: Optional[str] = None
):
    """
    plots points (array samples*2) and its oppositions (array samples*2)

    test:
        p = np.array([
            [1, 3],
            [4, 6],
            [9, 7]
        ])

        op = np.array([
            [3, 1],
            [5, 6],
            [4, 8]
        ])

        bounds = np.array([
            [-1, 10],
            [0, 11]
        ])

        plot_opposition(p, op, bounds, title="dfsfwe", net=True)
    """
    import matplotlib.pyplot as plt

    xmin, xmax, ymin, ymax = _set_axis(bounds)

    if net:
        _plot_net(xmin, xmax, ymin, ymax)

    for i in range(points.shape[0]):
        p1 = points[i, :]
        p2 = oppositions[i, :]

        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color='blue')

    plt.plot(points[:, 0], points[:, 1],
             'o', color='green', label='current points', markeredgecolor="black")
    plt.plot(oppositions[:, 0], oppositions[:, 1],
             'o', color='red', label='points oppositions',
             markeredgecolor="black")

    plt.annotate('center of zone', xy=((xmin + xmax) / 2, (ymin + ymax) / 2),
                 xytext=_random_coords(xmin, xmax, ymin, ymax, wd=0.4),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 # color = 'red',
                 bbox=dict(boxstyle="round", fc="0.8")  # , arrowprops = arrowprops
                 )

    plt.legend()
    plt.title(title)

    _save_fig(save_as)

    plt.show()


def plot_oppositions(
    point: Tuple[float, float],
    oppositions: np.ndarray,
    bounds: Tuple[
        Tuple[float, float],
        Tuple[float, float]
    ],
    names: Sequence[str],
    title: str,
    net: bool = False,
    save_as: Optional[str] = None
):
    """
    plots several opposition points of the point

    test:
        p = np.array([1, 3])

        op = np.array([
            [3, 1],
            [5, 6],
            [4, 8]
        ])

        bounds = np.array([
            [-1, 10],
            [0, 11]
        ])

        names = ['1', '2', '3']

        plot_oppositions(p, op, bounds, names, title = "opop", net = True)
    """

    import matplotlib.pyplot as plt

    xmin, xmax, ymin, ymax = _set_axis(bounds)

    if net:
        _plot_net(xmin, xmax, ymin, ymax)

    xg = (xmax - xmin)
    yg = (ymax - ymin)

    for i in range(oppositions.shape[0]):
        p1 = point
        p2 = oppositions[i, :]

        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color='blue')
        plt.annotate(names[i], (p2[0], p2[1]),
                     xytext=(p2[0] + np.random.uniform(-1, 1) * 0.2 * xg,
                             p2[1] + np.random.uniform(-1, 1) * 0.1 * yg),
                     arrowprops=dict(arrowstyle="fancy"),
                     bbox=dict(boxstyle="round", fc="w"))

    plt.plot(point[0], point[1], 'o', color='green', label='current point', markeredgecolor="black")
    plt.plot(oppositions[:, 0], oppositions[:, 1], 'o', color='red', label='points oppositions',
             markeredgecolor="black")

    plt.legend()
    plt.title(title)

    _save_fig(save_as)

    plt.show()


def plot_pop(
    points: np.ndarray,
    bounds: Tuple[
        Tuple[float, float],
        Tuple[float, float]
    ],
    title: str,
    net: bool = False,
    save_as: Optional[str] = None
):
    """
    plots points from 2D array with shape samples*2
    """
    import matplotlib.pyplot as plt

    xmin, xmax, ymin, ymax = _set_axis(bounds)

    if net:
        _plot_net(xmin, xmax, ymin, ymax)

    plt.plot(points[:, 0], points[:, 1], 'o', color='green', label='population points', markeredgecolor="black")

    # plt.legend()
    plt.title(title)

    _save_fig(save_as)

    plt.show()


_colors = (
    'green',
    'red',
    'blue',
    'yellow',
    'black',
    'orange',
    'violet'
)


def plot_pop_op(
    points,
    bounds: Tuple[
        Tuple[float, float],
        Tuple[float, float]
    ],
    names: List[str],
    title: str,
    net: bool = False,
    save_as: Optional[str] = None
):
    import matplotlib.pyplot as plt

    xmin, xmax, ymin, ymax = _set_axis(bounds)

    if net:
        _plot_net(xmin, xmax, ymin, ymax)

    names = ['start random pop'] + names

    s = int(points.shape[0] / len(names))
    assert s > 0, 'not enough points'

    for i, name in enumerate(names):
        pt = points[i * s:(i + 1) * s]
        plt.plot(
            pt[:, 0], pt[:, 1], 'o',
            color=_colors[i % len(_colors)],
            label=name, markeredgecolor="black"
        )

    plt.legend()
    plt.title(title)

    _save_fig(save_as)

    plt.show()




