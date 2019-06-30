import numpy as np


def add_half_spiral_2d(factory, fst, lst, nturns, rad):
    # Center points of conductors
    xs, ys, zs = [np.concatenate([np.linspace(fc, lc, nturns)], axis=0)
                  for fc, lc in zip(fst,lst)]     
    # generate the disks
    disktags = [factory.addDisk(xc, yc, zc, rad, rad) 
                for xc,yc,zc in zip(xs, ys, zs)]

    return disktags    
    