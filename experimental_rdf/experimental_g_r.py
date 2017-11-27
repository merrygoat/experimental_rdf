#!/usr/bin/env python
import os
import scipy as sp
from scipy import spatial,stats
import numpy as np
from numpy import random
import sys
import time

start_time = time.time()
np.seterr(divide='ignore', invalid='ignore')
filename= ''

print("#----------------------------------------------------------------")
print("#-----------Tool for determination of G(r)-----------------------")
print("#----------------------------------------------------------------")

if len(sys.argv)<3:
    print("ERROR: missing mandatory argument!")
    print("usage: Gr coordfile  binsize[pixel]")
    print("example SeriesXX_coords.txt 0.5")
    exit(11)
if len(sys.argv) >= 3:
  filename= sys.argv[1]
  dr = float(sys.argv[2])



print("# file %s"%filename)
C= np.genfromtxt(filename)


print("# size before cutting borders: ")

print("# x size: %d %d "%(C[:,0].min(),C[:,0].max()))
print("# y size: %d %d "%(C[:,1].min(),C[:,1].max()))
print("# z size: %d %d "%(C[:,2].min(),C[:,2].max()))




num_particles=len(C)



print("# Number of Particles: %d "%num_particles)
print("# init done %1.3f s"%(time.time() - start_time))
start_time = time.time()
bins=np.arange(0,C[:,0].max(),dr)

# for i in range(3):
# print i
    #create random numbers in the same range as the real particles
ID0=np.random.uniform(C[:,0].min(),C[:,0].max(),size=len(C))
ID1=np.random.uniform(C[:,1].min(),C[:,1].max(),size=len(C))
ID2=np.random.uniform(C[:,2].min(),C[:,2].max(),size=len(C))


ID=np.vstack((ID0,ID1))
ID=(np.vstack((ID,ID2))).T


#calculate distances
RID=sp.spatial.distance.pdist(ID, 'euclidean').flatten()

HID,binsID,binnumbersID=sp.stats.binned_statistic(RID, RID, statistic='count', bins=bins)
#     try:
#         IdealPart+=HID
#     except Exception, e:
#         IdealPart=HID


# HID=IdealPart/3.


print("# ideal gas done %1.3f s"%(time.time() - start_time))
start_time = time.time()

RC=sp.spatial.distance.pdist(C, 'euclidean').flatten()

print("# particles done %1.3f s"%(time.time() - start_time))
start_time = time.time()



print("# bin size: %f array [%f,%f]"%(dr,1,RC.max()))
print("# number of bins: %d"%(len(bins)))

H,bins,binnumbers=sp.stats.binned_statistic(RC, RC, statistic='count', bins=bins)

#calculate bin centers and divide ParticleHist/IdealGasHist
bincenters = 0.5*(bins[1:]+bins[:-1])
hist=H/HID
#take care of 0/0 and x/0, set to 0
hist[np.isnan(hist)] = 0
hist[np.isinf(hist)] = 0
#save result in file
f=open(filename+"_CC_r.hist",'wb')
np.savetxt(f, np.column_stack((bincenters,hist,H,HID)), fmt='%f')

f.close()

print("# binning done %1.3f s"%(time.time() - start_time))
import pylab as pl
pl.plot(bincenters[:len(bincenters)],hist[:len(bincenters)],'-')
pl.ylim(0,5)
pl.xlim(0,150)
pl.show()
# exit(0)
