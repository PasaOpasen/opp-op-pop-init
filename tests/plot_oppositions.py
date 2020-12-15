
import random
import numpy as np
import matplotlib.pyplot as plt 



def plot_oppositions(point, oppositions, bounds, names, title, net = False, save_as = None):
    
    xmin, xmax, ymin, ymax = bounds[0,0],bounds[0,1], bounds[1,0], bounds[1,1]
    plt.axis((xmin, xmax, ymin, ymax))

    if net:
        xs = np.arange(round(xmin), round(xmax) + 1)
        ys = np.arange(round(ymin), round(ymax) + 1)

        for xc in xs:
            plt.axvline(x=xc, color='k', linestyle='--', lw = 0.3)
        
        for yc in ys:
            plt.axhline(y=yc, color='k', linestyle='--', lw = 0.3)

    xg = (xmax-xmin)
    yg = (ymax-ymin)    

    for i in range(oppositions.shape[0]):

        p1 = point
        p2 = oppositions[i, :]
    
        plt.plot([p1[0],p2[0]], [p1[1],p2[1]], 'r--', color = 'blue')
        plt.annotate(names[i], (p2[0],p2[1]), 
            xytext=(p2[0]+np.random.uniform(-1,1)*0.2*xg, p2[1]+np.random.uniform(-1,1)*0.1*yg),
                     arrowprops = dict(arrowstyle="fancy"),
                     bbox=dict(boxstyle="round", fc="w"))

    plt.plot(point[0], point[1], 'ro', color = 'green', label = 'current point', markeredgecolor = "black")
    plt.plot(oppositions[:, 0], oppositions[:, 1], 'ro', color = 'red', label = 'points oppositions', markeredgecolor = "black")

    
    def random_coords(xmin, xmax, ymin, ymax, wd = 0.2):
        
        rx, ry = tuple(np.random.uniform(0.5-wd, 0.5+wd, 2))
        
        return (xmin*rx + (1-rx)*xmax, ymin*ry + (1-ry)*ymax)

    #plt.annotate('center of zone', xy = ((xmin+xmax)/2, (ymin+ymax)/2), 
    #             xytext = random_coords(xmin, xmax, ymin, ymax, wd = 0.3) ,
    #         arrowprops=dict(facecolor='red', shrink=0.05), 
    #          #color = 'red',
    #          bbox = dict(boxstyle ="round", fc ="0.8")#, arrowprops = arrowprops
    #         )


    plt.legend()
    plt.title(title)

    if save_as != None:
        plt.savefig(save_as, dpi = 200)
    
    plt.show()
        






if __name__ == '__main__':
    
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