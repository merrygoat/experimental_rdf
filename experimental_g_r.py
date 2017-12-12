#!/usr/bin/env python
import scipy as sp
from scipy import spatial,stats
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from tqdm import trange


def read_xyz_file(filename, dimensions):
    particle_positions = []
    frame_number = 0
    line_number = 0
    with open(filename, 'r') as input_file:
        for line in input_file:
            if line_number == 0:
                # Check for blank line at end of file
                if line != "":
                    frame_particles = int(line)
                    particle_positions.append(np.zeros((frame_particles, dimensions)))
            elif line_number == 1:
                comment = line
            else:
                particle_positions[frame_number][line_number-2] = line.split()[1:]
            line_number += 1
            # If we have reached the last particle in the frame, reset counter for next frame
            if line_number == (frame_particles + 2):
                line_number = 0
                frame_number += 1

    num_frames = len(particle_positions)

    return particle_positions, num_frames


def get_list_max(frame_list):
    list_max = 0
    for frame in frame_list:
        if frame.max() > list_max:
            list_max = frame.max()
    return list_max


def main():
    start_time = time.time()

    # Read in and process arguments
    if len(sys.argv) != 5:
        print("ERROR: missing mandatory argument!")
        print("usage: Gr coordfile  binsize[pixel] dimensions number_of_frames_to_analyze")
        print("example SeriesXX_coords.txt 0.5 3 100")
        exit(2)
    if len(sys.argv) == 5:
        filename = sys.argv[1]
        bin_width = float(sys.argv[2])
        dimensions = int(sys.argv[3])
        user_frames = int(sys.argv[4])

    particle_positions, num_frames = read_xyz_file(filename, dimensions)
    if user_frames < num_frames:
        num_frames = user_frames

    # Calculate bins and bin centers
    bins = np.arange(0, get_list_max(particle_positions), bin_width)
    bincenters = 0.5 * (bins[1:] + bins[:-1])

    num_particles = np.zeros(num_frames, dtype=np.int)
    hist = np.zeros((num_frames, len(bins)-1))
    # calculate the g(r) for each frame
    for frame in trange(num_frames):
        num_particles[frame] = len(particle_positions[frame])

        # create a random gas in the same box as the real particles
        rg_positions = np.zeros((num_particles[frame], dimensions))
        for i in range(0, dimensions):
            rg_positions[:, i] = np.random.uniform(particle_positions[frame][:, i].min(),
                                                   particle_positions[frame][:, i].max(), size=num_particles[frame])

        # calculate distances for random gas and bin them
        rg_distances = sp.spatial.distance.pdist(rg_positions, 'euclidean').flatten()
        rg_hist = sp.stats.binned_statistic(rg_distances, rg_distances, statistic='count', bins=bins)[0]

        # Calculate distances for experimental particles and bin them
        exp_distances = sp.spatial.distance.pdist(particle_positions[frame], 'euclidean').flatten()
        exp_hist = sp.stats.binned_statistic(exp_distances, exp_distances, statistic='count', bins=bins)[0]

        # normalisation of g(r)
        hist[frame] = exp_hist / rg_hist
        # take care of 0/0 and x/0, set to 0
        hist[frame][np.isnan(hist[frame])] = 0
        hist[frame][np.isinf(hist[frame])] = 0

    # normalisation of frames
    hist = np.sum((hist.T * num_particles/np.sum(num_particles)).T, axis=0)

    # save result in file
    with open("%s_hist_bin%1.1f.txt" % (filename, bin_width), 'wb') as f:
        np.savetxt(f, np.column_stack((bincenters, hist)), fmt='%f')

    plt.plot(bincenters[:len(bincenters)], hist[:len(bincenters)], '-')
    plt.ylim(0, 5)
    plt.xlim(0, 150)
    plt.show()

main()
