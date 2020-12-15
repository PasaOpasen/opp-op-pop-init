import numpy as np
import matplotlib.pyplot as plt 



def plot_opposition(points, oppositions, bounds, title, net = False, save_as = None):
    
    xmin, xmax, ymin, ymax = bounds[0,0],bounds[0,1], bounds[1,0], bounds[1,1]
    plt.axis((xmin, xmax, ymin, ymax))

    if net:
        xs = np.arange(round(xmin), round(xmax) + 1)
        ys = np.arange(round(ymin), round(ymax) + 1)

        for xc in xs:
            plt.axvline(x=xc, color='k', linestyle='--', lw = 0.3)
        
        for yc in ys:
            plt.axhline(y=yc, color='k', linestyle='--', lw = 0.3)

    
    for i in range(points.shape[0]):

        p1 = points[i, :]
        p2 = oppositions[i, :]
    
        plt.plot([p1[0],p2[0]], [p1[1],p2[1]], 'r--', color = 'blue')

    plt.plot(points[:, 0], points[:, 1], 'ro', color = 'green', label = 'current points', markeredgecolor = "black")
    plt.plot(oppositions[:, 0], oppositions[:, 1], 'ro', color = 'red', label = 'points oppositions', markeredgecolor = "black")

    
    def random_coords(xmin, xmax, ymin, ymax, wd = 0.2):
        
        rx, ry = tuple(np.random.uniform(0.5-wd, 0.5+wd, 2))
        
        return (xmin*rx + (1-rx)*xmax, ymin*ry + (1-ry)*ymax)

    plt.annotate('center of zone', xy = ((xmin+xmax)/2, (ymin+ymax)/2), 
                 xytext = random_coords(xmin, xmax, ymin, ymax, wd = 0.4) ,
             arrowprops=dict(facecolor='red', shrink=0.05), 
              #color = 'red',
              bbox = dict(boxstyle ="round", fc ="0.8")#, arrowprops = arrowprops
             )


    plt.legend()
    plt.title(title)

    if save_as != None:
        plt.savefig(save_as, dpi = 200)
    
    plt.show()
        






if __name__ == '__main__':
    
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

    plot_opposition(p, op, bounds, title = "dfsfwe", net = True)
