
import numpy as np
import matplotlib.pyplot as plt 


colors = [
    'green',
    'red',
    'blue',
    'yellow',
    'black',
    'orange'
]


def plot_pop_op(points, names, bounds, title, net = False, save_as = None):
    
    xmin, xmax, ymin, ymax = bounds[0,0], bounds[0,1], bounds[1,0], bounds[1,1]
    plt.axis((xmin, xmax, ymin, ymax))

    if net:
        xs = np.arange(round(xmin), round(xmax) + 1)
        ys = np.arange(round(ymin), round(ymax) + 1)

        for xc in xs:
            plt.axvline(x=xc, color='k', linestyle='--', lw = 0.3)
        
        for yc in ys:
            plt.axhline(y=yc, color='k', linestyle='--', lw = 0.3)

    names = ['start random pop'] + names

    s = int(points.shape[0]/len(names))

    for i, name in enumerate(names):
        pt = points[i*s:(i+1)*s, :]
        plt.plot(pt[:, 0], pt[:, 1], 'ro', 
        color = colors[i], 
        label = name, markeredgecolor = "black")


    plt.legend()
    plt.title(title)

    if save_as != None:
        plt.savefig(save_as, dpi = 200)
    
    plt.show()