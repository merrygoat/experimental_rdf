#!/usr/bin/env python
import scipy as sp
from scipy import spatial,stats
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

def main():
    start_time = time.time()
    #np.seterr(divide='ignore', invalid='ignore')

    # Read in and process arguments
    if len(sys.argv) != 3:
        print("ERROR: missing mandatory argument!")
        print("usage: Gr coordfile  binsize[pixel]")
        print("example SeriesXX_coords.txt 0.5")
        exit(2)
    if len(sys.argv) == 3:
        filename = sys.argv[1]
        dr = float(sys.argv[2])

    print("# file %s" % filename)
    particle_positions = np.genfromtxt(filename)
    num_particles = len(particle_positions)

    print("# size before cutting borders: ")
    print("# x size: %d %d "%(particle_positions[:,0].min(),particle_positions[:,0].max()))
    print("# y size: %d %d "%(particle_positions[:,1].min(),particle_positions[:,1].max()))
    print("# z size: %d %d "%(particle_positions[:,2].min(),particle_positions[:,2].max()))
    print("# Number of Particles: %d " % num_particles)
    print("# init done %1.3f s" % (time.time() - start_time))

    start_time = time.time()
    bins=np.arange(0,particle_positions[:,0].max(), dr)
    dimensions = 3

    # create a random gas in the same box as the real particles
    random_gas_positions = np.zeros((num_particles, dimensions))
    for i in range(0, dimensions):
        random_gas_positions[:, i] = np.random.uniform(particle_positions[:,i].min(),particle_positions[:,i].max(),size=num_particles)

    #calculate distances for randomn gas and bin them
    RID = sp.spatial.distance.pdist(random_gas_positions, 'euclidean').flatten()
    HID, binsID, binnumbersID = sp.stats.binned_statistic(RID, RID, statistic='count', bins=bins)
    print("# ideal gas done %1.3f s"%(time.time() - start_time))

    # Calculate distances for experimental particles and bin them
    start_time = time.time()
    RC = sp.spatial.distance.pdist(particle_positions, 'euclidean').flatten()
    print("# particles done %1.3f s"%(time.time() - start_time))
    print("# bin size: %f array [%f,%f]"%(dr,1,RC.max()))
    print("# number of bins: %d"%(len(bins)))
    H,bins,binnumbers=sp.stats.binned_statistic(RC, RC, statistic='count', bins=bins)

    #calculate bin centers and normalise
    bincenters = 0.5*(bins[1:]+bins[:-1])
    hist=H/HID
    #take care of 0/0 and x/0, set to 0
    hist[np.isnan(hist)] = 0
    hist[np.isinf(hist)] = 0
    #save result in file
    with open(filename+"_CC_r.hist",'wb') as f:
        np.savetxt(f, np.column_stack((bincenters,hist,H,HID)), fmt='%f')

    print("# binning done %1.3f s"%(time.time() - start_time))

    plt.plot(bincenters[:len(bincenters)],hist[:len(bincenters)],'-')
    plt.ylim(0,5)
    plt.xlim(0,150)
    plt.show()

main()