"""Network analysis for the evolving system."""

import numpy as np
from classes import NetworkTimeseries, NetworkAverage


def run_analysis(
    data_dir: str,
    aver_window: int = 100,
    n_steps: int = 1000,
    spacing: str = 'geo',
    n_plots: int = 0,
    t_min: int = 1,
    c_size: int = 4,
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

    t_min : int = 1
        Minimum timestep to start the analysis from.

    c_size : int = 4
        Critical size dividing "short" and "long" polymers (critical
        nucleation size). 
    """
    test_nts = NetworkTimeseries(
        data_directory=data_dir,
        n_steps=n_steps,
        spacing=spacing,
        aver_window=aver_window,
        n_plots=n_plots,
        t_min=t_min,
        critical_size=c_size,
    )

    # test_nts.time_series[-1].print_norm_graph()

    np.savetxt("output_data/timesteps.txt", test_nts.time_steps)

    n_nodes = test_nts.print_number_of_nodes()
    np.savetxt("output_data/n_nodes.txt", n_nodes)

    degrees, in_degrees_std, out_degrees_std = test_nts.compute_mean_deg()
    array_deg = np.array([degrees, in_degrees_std, out_degrees_std])
    np.save("output_data/degrees.npy", array_deg)

    deg_small, deg_large = test_nts.compute_partial_deg(critical_size=c_size)
    array_partials = np.array([deg_small, deg_large])
    np.save("output_data/partial_degrees.npy", array_partials)

    stren, in_stren_std, out_stren_std = test_nts.compute_mean_str()
    array_str = np.array([stren, in_stren_std, out_stren_std])
    np.save("output_data/strength.npy", array_str)

    clust_coeff, cc_small, cc_large = test_nts.compute_mean_cc(
        critical_size=c_size)
    np.save("output_data/clust_coeff.npy", clust_coeff)
    array_partials = np.array([cc_small, cc_large])
    np.save("output_data/partial_cc.npy", array_partials)

    diameter, mean_dist = test_nts.compute_distances()
    np.save("output_data/distances.npy", np.array([diameter, mean_dist]))

    matrix_dist = test_nts.compute_matrix_dist()
    np.save("output_data/matrix_dist.npy", matrix_dist)


def run_equi_dist_analysis(
    data_dir: str,
    t_min: int = 1,
):
    """
    Computes the equilibrium distributions and prints the results.

    Arguments
    ---------

    data_dir : str
        The file with the input data. File has to contain a matrix (N, T + 1),
        where for every monomer is indicated the size of the polymer it is
        contained in, at every simulation frame.

    t_min : int = 1
        Minimum timestep to start the analysis from.
    """
    test_nts = NetworkAverage(
        data_directory=data_dir,
        t_min=t_min,
    )

    test_nts.get_deg_centrality_distribution("output_figures/deg_centrality")
    test_nts.get_h_index_centrality_distribution(
        "output_figures/h_index_centrality"
    )
