"""Run analysis in folder coop_1."""

import sys

# Not a Pypi package, modify as nedded
sys.path.append('../interaction_network/src/')

from main import run_analysis

DATA_DIR = ("/Users/mattebecchi/00_graph_analysis/coop_1/"
    "combined_data.npy") # The directory of yhe input file

run_analysis(
    data_dir=DATA_DIR,
    aver_window=100,    # Average the graph over this many frames
    n_steps=1000,       # Sampling of the trajectory
    spacing='geo',      # 'geo' or 'lin' spacing of the smapled frames
    n_plots=0,          # Plot these graphs
)
