"""Functions for creating the Networsk from specific systems."""

from typing import List
import numpy as np


def read_hex(
    labels: np.ndarray,
    aver_window: int,
    selected_time_steps: List[int],
):
    """Read the data from the hexagon simulations of Martina and Claudio.

    Parameters
    ----------

    labels : np.ndarray of shape (n_particles, n_frames)
        labels[i][t] is the size of the polymer monomer i is part of at
        timestep t.

    aver_window : int
        The graph is constructed averaging over the last "aver_window"
        timesteps, for increasing stability.

    selected_time_steps : List[int]
        The list of timesteps at which the graphs will be computed.

    Returns
    -------

    max_size : int
        The maximum polymer length in the simulation.

    matrix_list : List[ndarray] of shape (n_nodes, n_nodes)
        The element i, j counts the occurences of a monomer moving from a
        construct of size power[i] to a construct of size power[j] in a single
        timestep. Self-edges are removed (otherwise they are too dominant).
    """
    labels = labels.astype(int)
    max_size = np.max(labels)
    matrix_list = []

    for time in selected_time_steps:
        edges_matrix = np.zeros((max_size, max_size))
        start_time = max(0, time - aver_window)
        num_of_frames = time - start_time

        # Slice the labels array for the relevant time window
        window_labels = labels[:, start_time:time + 1]

        # Calculate transitions for each particle
        l_0 = window_labels[:, :-1].flatten()
        l_1 = window_labels[:, 1:].flatten()

        # Create transition counts matrix
        transition_counts = np.zeros((max_size, max_size), dtype=int)
        np.add.at(transition_counts, (l_0 - 1, l_1 - 1), 1)

        # Average the transition matrix
        edges_matrix = transition_counts / num_of_frames

        # Remove self-edges
        for i, _ in enumerate(edges_matrix):
            edges_matrix[i][i] = 0.0

        matrix_list.append(edges_matrix)

    return max_size, matrix_list
