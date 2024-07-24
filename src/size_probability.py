"""Compute the average size a monomer belongs to."""

import numpy as np
import matplotlib.pyplot as plt


T_CONV = 0.3
SIZE_DIR_COOP = ("/Users/mattebecchi/00_graph_analysis/coop_1/"
    "combined_data.npy")
TIME_DIR_COOP = ("/Users/mattebecchi/00_graph_analysis/coop_1/"
    "output_data/timesteps.txt")
NODES_DIR_COOP = ("/Users/mattebecchi/00_graph_analysis/coop_1/"
    "output_data/n_nodes.txt")
SIZE_DIR_ISO = ("/Users/mattebecchi/00_graph_analysis/iso_1/"
    "combined_data.npy")
TIME_DIR_ISO = ("/Users/mattebecchi/00_graph_analysis/iso_1/"
    "output_data/timesteps.txt")
NODES_DIR_ISO = ("/Users/mattebecchi/00_graph_analysis/iso_1/"
    "output_data/n_nodes.txt")

time_coop = np.array([int(t) for t in np.loadtxt(TIME_DIR_COOP)])
time_iso = np.array([int(t) for t in np.loadtxt(TIME_DIR_ISO)])

### Compute and plot the average polymer size ###
tmp = np.load(SIZE_DIR_COOP)
sizes_coop = tmp[:, 1:].T
mean_size_coop = np.mean(sizes_coop, axis=0)
tmp = np.load(SIZE_DIR_ISO)
sizes_iso = tmp[:, 1:].T
mean_size_iso = np.mean(sizes_iso, axis=0)

fig1, ax = plt.subplots()
fig1.suptitle("- Note: size is averaged with the number of monomers -")
ax.plot(time_coop * T_CONV, mean_size_coop[time_coop], label='Coop')
ax.plot(time_iso * T_CONV, mean_size_iso[time_iso], label='Iso')
ax.set_xlabel("Simulation time [ns]")
ax.set_ylabel("Average polymer size")
ax.set_xscale('log')
ax.legend()
fig1.savefig("final_figures/polymer_size.png", dpi=600)

### Average polymer size vs number of nodes in the network ###
n_nodes_coop = np.loadtxt(NODES_DIR_COOP)
n_nodes_iso = np.loadtxt(NODES_DIR_ISO)

fig2, ax = plt.subplots()
fig2.suptitle("- Note: size is averaged with the number of monomers -")
ax.plot(n_nodes_coop, mean_size_coop[time_coop], label='Coop')
ax.plot(n_nodes_iso, mean_size_iso[time_iso], label='Iso')
ax.set_xlabel("Number of nodes")
ax.set_ylabel("Average polymer size")
ax.legend()
fig2.savefig("final_figures/size_vs_nodes.png", dpi=600)

plt.show()
