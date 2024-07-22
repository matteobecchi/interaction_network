"""Network analysis for the evolving system."""

import numpy as np
from classes import NetworkTimeseries

CC_THR = 0.0 # Threshold for the clustering coefficient measure


def run_analysis(
    data_dir: str,
    aver_window: int = 100,
    n_steps: int = 1000,
    spacing: str = 'geo',
    n_plots: int = 0,
    t_min: int = 1,
):
    """
    Runs the analysis and print the results.

    Arguments
    ---------

    data_dir : str
        The file with the input data. File has to contain a matrix (N, T + 1),
        where for every monomer is indicated the size of the polymer it is
        contained in, at every simulation frame.

    aver_window : int = 100
        The number of frames over which the networks are averaged.

    n_steps : int = 1000
        Sampling of the trajectory.

    spacing : str = 'geo'
        Spacing of the frames, can be 'geo' or 'lin'.

    n_plots : int = 0
        Number of frames at which the gaphs are plotted.
    """
    test_nts = NetworkTimeseries(
        data_directory=data_dir,
        n_steps=n_steps,
        spacing=spacing,
        aver_window=aver_window,
        n_plots=n_plots,
        t_min=t_min,
    )

    # test_nts.time_series[-1].print_norm_graph()

    np.savetxt("output_data/timesteps.txt", test_nts.time_steps)

    n_nodes = test_nts.print_number_of_nodes()
    np.savetxt("output_data/n_nodes.txt", n_nodes)

    degrees, in_degrees_std, out_degrees_std = test_nts.compute_mean_deg()
    array_deg = np.array([degrees, in_degrees_std, out_degrees_std])
    np.save("output_data/degrees.npy", array_deg)

    deg_small, deg_large = test_nts.compute_partial_deg(critical_size=4)
    array_partials = np.array([deg_small, deg_large])
    np.save("output_data/partial_degrees.npy", array_partials)

    stren, in_stren_std, out_stren_std = test_nts.compute_mean_str()
    array_str = np.array([stren, in_stren_std, out_stren_std])
    np.save("output_data/strength.npy", array_str)

    clust_coeff, cc_small, cc_large = test_nts.compute_mean_cc(
        CC_THR, critical_size=4)
    np.save("output_data/clust_coeff.npy", clust_coeff)
    array_partials = np.array([cc_small, cc_large])
    np.save("output_data/partial_cc.npy", array_partials)

    diameter, mean_dist = test_nts.compute_distances()
    np.save("output_data/distances.npy", np.array([diameter, mean_dist]))
