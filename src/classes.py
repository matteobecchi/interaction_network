"""Objects for studing networks of interactions."""

import copy
from typing import List, Tuple
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import rgb2hex
from input_functions import read_hex


class Network:
    """Stores the network of exchanges between two timesteps.

    Parameters
    ----------

    n_particles : int
        The number of particles in the simulation.

    Attributes
    ----------

    _n_nodes : int
        The number of graph nodes.

    _edges : ndarray of float of shape (n_nodes, n_nodes)
        The weighted edges of the graph.

    _norm_edges : ndarray of float of shape (n_nodes, n_nodes)
        The weighted edges of the graph, normalized so that they sum equals 1.

    _time : int, default -1.
        The timestep to which the graph corresponds.

    _labels : List[str]
        The labels for the graph nodes.
    """
    def __init__(
        self,
        time: int,
        edges_matrix: np.ndarray,
        max_size: int,
    ):
        self._time = time
        self._edges = edges_matrix
        self._norm_edges = edges_matrix / (np.sum(self._edges))
        self._labels = np.arange(1, max_size + 1)

        # Compute the number of non-isolated nodes
        n_nodes = 0
        for i, row in enumerate(self._edges):
            if np.sum(row) > 0.0 or np.sum(self._edges[:, i]) > 0.0:
                n_nodes += 1
        self._n_nodes = n_nodes


    def get_n_nodes(self) -> int:
        """Returns the number of nodes in the graph."""
        return self._n_nodes


    def get_edges(self) -> np.ndarray:
        """Retruns a copy of the graph edges.

        Returns
        -------
        ndarray of float of shape (n_nodes, n_nodes)
            The weighted edges of the graph.
        """
        return copy.deepcopy(self._edges)


    def print_graph(self):
        """Prints the graph in a matrix form."""
        print(self._edges)


    def print_norm_graph(self):
        """Prints the normalize graph in a matrix form."""
        np.set_printoptions(precision=2, suppress=False)
        print(self._norm_edges*100)


    def mean_str(self) -> Tuple[float, float, float]:
        """Computes the average strength of the graph.

        Returns
        -------

        mean_str : float
            The mean of the nodes' strength in the graph. Notice that the mean
            in-strength and out-strength are the same, so one number is
            enough.

        std_in_str : float
            The standard deviation of the nodes' in-strength in the graph.

        std_out_str : float
            The standard deviation of the nodes' out-strength in the graph.
        """
        in_str, out_str = [], []
        for i in range(len(self._labels)):
            tmp_in, tmp_out = 0.0, 0.0
            for j in range(len(self._labels)):
                if j != i:
                    tmp_in += self._edges[i][j]
                    tmp_out += self._edges[j][i]
            if tmp_in > 0.0 or tmp_out > 0.0:
                in_str.append(tmp_in)
                out_str.append(tmp_out)

        if (np.mean(in_str) - np.mean(out_str)) / np.mean(in_str) > 0.01:
            print("Warning: in- and out- mean strength are different: "
                f"{np.mean(in_str)} != {np.mean(out_str)}")

        mean_str = np.mean(in_str)
        std_in_str = np.std(in_str)
        std_out_str = np.std(out_str)

        return mean_str, std_in_str, std_out_str


    def mean_deg(self) -> Tuple[float, float, float]:
        """Computes the average degree of the graph.

        Returns
        -------

        mean_deg : float
            The mean of the nodes' degree in the graph. Notice that the mean
            in-degree and out-degree are the same, so one number is enough.

        std_in_deg : float
            The standard deviation of the nodes' in-degree in the graph.

        std_out_deg : float
            The standard deviation of the nodes' out-degree in the graph.
        """
        in_deg, out_deg = [], []
        for i in range(len(self._labels)):
            tmp_in, tmp_out = 0, 0
            for j in range(len(self._labels)):
                if self._edges[i][j] > 0 and j != i:
                    tmp_in += 1
                if self._edges[j][i] > 0 and j != i:
                    tmp_out += 1
            if tmp_in > 0 or tmp_out > 0:
                in_deg.append(tmp_in)
                out_deg.append(tmp_out)

        if np.mean(in_deg) != np.mean(out_deg):
            print("Warning: in- and out- mean degrees are different: "
                f"{np.mean(in_deg)} != {np.mean(out_deg)}")

        mean_deg = np.mean(in_deg)
        std_in_deg = np.std(in_deg)
        std_out_deg = np.std(out_deg)

        return mean_deg, std_in_deg, std_out_deg


    def partial_mean_deg(
        self,
        critical_size: int,
    ) -> Tuple[float, float]:
        """Computes the partial (undirected) degrees of the graph.

        Arguments
        ---------

        critical_size : int
            Distinguish between the mean degree for the sizes below and
            above the critical_size.

        Returns
        -------

        mean_deg_small : float
            The mean of the nodes' degree in the graph, for the sizes
            smaller than the critical size.

        mean_deg_large : float
            The mean of the nodes' degree in the graph, for the sizes
            larger than the critical size.
        """
        deg_small, deg_large = [], []
        for i in range(len(self._labels)):
            tmp = 0
            for j in range(len(self._labels)):
                if j != i:
                    # +0.5 so that it is consistent with the directed averages
                    if self._edges[i][j] > 0:
                        tmp += 0.5
                    if self._edges[j][i] > 0:
                        tmp += 0.5
            if tmp > 0 and i <= critical_size - 1:
                deg_small.append(tmp)
            elif tmp > 0 and i > critical_size - 1:
                deg_large.append(tmp)

        if len(deg_small) > 0:
            mean_deg_small = np.mean(deg_small)
        else:
            mean_deg_small = 0.0

        if len(deg_large) > 0:
            mean_deg_large = np.mean(deg_large)
        else:
            mean_deg_large = 0.0

        return mean_deg_small, mean_deg_large


    def mean_cc(
        self,
        thr: float,
        critical_size: int,
    ) -> Tuple[float, float, float]:
        """Computes the average clustering coefficient.

        Parameters
        ----------

        thr : float
            In computing the clustering coefficinet, only consider a link if
            its weigth is higher than thr.

        critical_size : int
            Distinguish between the mean clustering coefficient for the sizes
            below and above the critical_size.

        Returns
        -------

        mean_cl_coeff : float
            The average clustering coefficient of the graph (it's a number
            between 0 -tree like graph- and 1 -fully connected graph-).
        """
        clustering_coefficients, sizes = [], []

        sym_matrix = self._norm_edges + self._norm_edges.T

        for i, node in enumerate(sym_matrix):
            neighbors = [j for j, value in enumerate(node)
                if j != i and value > thr]
            degree = len(neighbors)

            if degree == 0:
                continue
            if degree == 1:
                clustering_coefficients.append(0.0)
                sizes.append(i + 1)
            else:
                total_triangles = 0
                for j, neigh_j in enumerate(neighbors):
                    for _, neigh_k in enumerate(neighbors[j + 1:]):
                        if sym_matrix[neigh_j][neigh_k] > thr:
                            total_triangles += 1

                clustering_coefficient = \
                    2.0 * total_triangles / (degree * (degree - 1))
                clustering_coefficients.append(clustering_coefficient)
                sizes.append(i + 1)

        if len(clustering_coefficients) > 0:
            mean_cl_coeff = np.mean(clustering_coefficients)
        else:
            mean_cl_coeff = 0.0

        cc_small_list = [value for i, value in enumerate(
            clustering_coefficients) if sizes[i] <= critical_size]
        cc_large_list = [value for i, value in enumerate(
            clustering_coefficients) if sizes[i] > critical_size]

        if len(cc_small_list) > 0:
            cc_small = np.mean(cc_small_list)
        else:
            cc_small = 0.0

        if len(cc_large_list):
            cc_large = np.mean(cc_large_list)
        else:
            cc_large = 0.0

        return mean_cl_coeff, cc_small, cc_large


    def dist_degrees(self, file_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the degree distribution of the graph.

        Returns
        -------

        deg_in : ndarray of shape (n_nodes,)
            The probability distribution of the nodes' in-degree.

        deg_out : ndarray of shape (n_nodes,)
            The probability distribution of the nodes' out-degree.
        """
        tot_n_nodes = self._edges.shape[0]
        deg_in, deg_out = np.zeros(tot_n_nodes), np.zeros(tot_n_nodes)
        for i, row in enumerate(self._edges):
            deg_in[i] = np.sum(self._edges.T[i] != 0)
            deg_out[i] = np.sum(row != 0)

        fig, axes = plt.subplots(2, 1, sharex=True)
        mask_in = deg_in > 0
        mask_out = deg_out > 0
        axes[0].hist(deg_in[mask_in], bins=int(max(deg_in)))
        axes[1].hist(deg_out[mask_out], bins=int(max(deg_out)))
        # axes[0].bar(range(1, tot_n_nodes + 1), deg_in)
        # axes[1].bar(range(1, tot_n_nodes + 1), deg_out)
        # axes[1].set_xlabel(r'Polymer size $i$')
        # axes[0].set_ylabel(r'In-degree $k_i^{in}$')
        # axes[1].set_ylabel(r'Out-degree $k_i^{out}$')
        # plt.show()
        fig.savefig(file_name, dpi=600)

        return deg_in, deg_out


    def dist_strengths(self, file_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the strength distribution of the graph.

        Returns
        -------

        str_in : ndarray of shape (n_nodes,)
            The probability distribution of the nodes' in-strength.

        str_out : ndarray of shape (n_nodes,)
            The probability distribution of the nodes' out-strength.
        """
        tot_n_nodes = self._edges.shape[0]
        str_in, str_out = np.zeros(tot_n_nodes), np.zeros(tot_n_nodes)
        for i, row in enumerate(self._edges):
            str_in[i] = np.sum(self._edges.T[i])
            str_out[i] = np.sum(row)

        fig, axes = plt.subplots(2, 1)
        axes[0].bar(range(1, tot_n_nodes + 1), str_in)
        axes[1].bar(range(1, tot_n_nodes + 1), str_out)
        axes[1].set_xlabel(r'Polymer size $i$')
        axes[0].set_ylabel(r'In-strength $s_i^{in}$')
        axes[1].set_ylabel(r'Out-strength $s_i^{out}$')
        # plt.show()
        fig.savefig(file_name, dpi=600)

        return str_in, str_out


    def dist_cc(self, thr) -> np.ndarray:
        """Computes the clustering coefficient distribution.

        Parameters
        ----------

        thr : float
            In computing the clustering coefficinet, only consider a link if
            its weigth is higher than thr.

        Returns
        -------

        clustering_coefficients : ndarray of float of shape (n_nodes,)
            The probability distribution of the nodes' cluastering
            coefficient.
        """
        clustering_coefficients = np.zeros(self._n_nodes)

        for node in range(self._n_nodes):
            neighbors = [i for i, value in enumerate(self._norm_edges[node])
                if value > 0 and i != node]

            if len(neighbors) < 2:
                clustering_coefficients[node] = 0.0
                continue

            total_triangles = 0
            for i, neigh_i in enumerate(neighbors):
                for _, neigh_j in enumerate(neighbors[i + 1:]):
                    if self._norm_edges[neigh_i][neigh_j] > thr:
                        total_triangles += 1

            degree = len(neighbors)

            clustering_coefficient = \
                2.0 * total_triangles / (degree * (degree - 1))
            clustering_coefficients[node] = clustering_coefficient

        return clustering_coefficients


    def plot(self, threshold: float, file_name: str):
        """Plots the graph using networkx.

        Parameters
        ----------

        threshold : float
            Ignore edges with a weight smaller than threshold.

        file_name : str
            The name of the file where the figure is saved as a .png.
        """
        edges = [
            (self._labels[i], self._labels[j],
                {'weight': self._edges[i][j]})
            for i in range(self._n_nodes) for j in range(self._n_nodes)
            if self._labels[i] != self._labels[j] and
            self._edges[i][j] > threshold
        ]

        graph = nx.DiGraph()
        graph.add_nodes_from(self._labels)# <--- ?
        graph.add_edges_from(edges)
        graph.remove_nodes_from(list(nx.isolates(graph)))
        edge_weights = [edge[2]['weight'] for edge in graph.edges(data=True)]
        rescale_width = np.max(edge_weights)

        cmap = plt.get_cmap("viridis", len(graph.nodes))
        palette = [rgb2hex(cmap(i)) for i in range(cmap.N)]

        fig, axes = plt.subplots()
        pos = nx.spring_layout(graph) # default is k=1/sqrt(n_nodes)
        nx.draw_networkx(graph,
            pos=pos,
            ax=axes,
            width=edge_weights/rescale_width*2,
            with_labels=True,
            node_size=200,
            node_color=palette,
        )
        plt.show()
        fig.savefig("output_figures/" + file_name + ".png", dpi=600)


    def bfs(self, start_node: int) -> dict:
        """Perform a breadth-first search from a node in the graph.

        Parameters
        ----------

        start_node : int
            The index of the node from which to start the search.

        Returns
        -------

        distances : dictionary
            Contains the minimum distance from the start_node to any
            other node.
        """
        distances = {node: float('inf') for node in range(len(self._labels))}
        distances[start_node] = 0

        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            for neigh, edge in enumerate(self._edges[current_node]):
                is_connected = edge != 0
                if is_connected and distances[neigh] == float('inf'):
                    distances[neigh] = distances[current_node] + 1
                    queue.append(neigh)

        return distances


    def diameter_and_average_distance(self)->Tuple[int, float]:
        """Compute the diameter and average node distance.

        Returns
        -------

        diameter : int
            The graph diameter.

        average_distance : float
            The mean distance between any pair of nodes.
        """
        diameter = 0
        total_distance_sum = 0
        counter = 0

        for node in range(len(self._labels)):
            distances = self.bfs(node)
            connect_dists = np.array(
                [d for d in distances.values() if d < float('inf')]
            )
            max_distance = max(connect_dists)
            diameter = max(diameter, max_distance)

            if np.any(connect_dists > 0):
                counter += len(connect_dists) - 1
            total_distance_sum += sum(connect_dists)

        if counter == 0.0:
            average_distance = 0.0
        else:
            average_distance = total_distance_sum / counter

        return diameter, average_distance


class NetworkTimeseries:
    """Stores the evolution of the interaction network.

    Parameters
    ----------

    data_directory : str
        The path to the input data file.

    t_min : int, default = 0
        At which timestep to start the analysis.

    t_max : int, default = -1
        At which timestep to stop the analysis. If -1, use the entire
        trajectory.

    n_steps : int, default = 1000
        How many graphs to compute, geometrically spaced along the
        trajectory.

    spacing : str = 'geo'
        Spacing of the frames, can be 'geo' or 'lin'.

    aver_window : int
        The graphs are constructed averaging over the last "aver_window"
        timesteps, for increasing stability.

    n_plots : int, default = 10
        How many graphs to plot.

    Arguments
    ---------

    time_steps : List[int]
        The number of frames at which the graphs are computed.

    time_series : List[Network]
        This is the list of the graphs for each timestep. All the time-
        dependent quantities are computed iterationg over this list.
    """

    def __init__(
        self,
        data_directory: str,
        t_min: int = 1,
        t_max: int = -1,
        n_steps: int = 1000,
        spacing: str = 'geo',
        aver_window: int = 100,
        n_plots: int = 10,
    ):
        data = np.load(data_directory)
        n_frames, n_particles = data.shape
        n_particles -= 1
        print(f"Particles: {n_particles}, frames: {n_frames}.")
        data = data[:, 1:].T

        if t_max == -1:
            t_max = n_frames - 1
            print(f"t_max = {t_max}")

        if spacing == 'geo':
            tmp_time_steps = np.geomspace(1, t_max - t_min + 1, num=n_steps,
                dtype=int)
            tmp_time_steps += t_min - 1
        elif spacing == 'lin':
            tmp_time_steps = np.linspace(t_min, t_max, num=n_steps, dtype=int)
        time_steps = [tmp_time_steps[0]]
        for i in range(1, len(tmp_time_steps)):
            if tmp_time_steps[i] != tmp_time_steps[i - 1]:
                time_steps.append(tmp_time_steps[i])
        plot_times = np.linspace(1, len(time_steps) - 1, num=n_plots,
            dtype=int)
        print("Plot at times: ", plot_times)

        self.time_steps = time_steps

        max_size, list_of_edges = read_hex(
            data,
            aver_window,
            time_steps,
        )

        self.time_series: List[Network] = []
        for i, edges_matrix in enumerate(list_of_edges):
            tmp_net = Network(time_steps[i], edges_matrix, max_size)
            self.time_series.append(tmp_net)

        for time in plot_times:
            self.time_series[time].plot(0.001, f"Fig0_{time}")


    def print_number_of_nodes(self) -> np.ndarray:
        """Print the number of nodes in the graph."""
        n_nodes = [graph.get_n_nodes() for graph in self.time_series]
        return np.array(n_nodes)


    def compute_mean_deg(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the mean graph degree for the time-series.

        Returns
        -------

        list_deg : np.ndarray of float of shape (n_steps,)
            The mean node degree at each timestep. Notice that the
            average in-degree and out-degree are the same, so only one
            number is necessary.

        list_in_std : np.ndarray of float of shape (n_steps,)
            The standard deviation of the in-degree distribution, at each
            timestep.

        list_out_std : np.ndarray of float of shape (n_steps,)
            The standard deviation of the out-degree distribution, at each
            timestep.
        """
        n_steps = len(self.time_series)
        list_deg = np.zeros(n_steps)
        list_in_std = np.zeros(n_steps)
        list_out_std = np.zeros(n_steps)

        for i, net in enumerate(self.time_series):
            deg, in_deg_std, out_deg_std = net.mean_deg()
            list_deg[i] = deg
            list_in_std[i] = in_deg_std
            list_out_std[i] = out_deg_std

        return list_deg, list_in_std, list_out_std


    def compute_partial_deg(
        self,
        critical_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the partial mean graph degree for the time-series.

        Arguments
        ---------

        critical_size : int
            Distinguish between the mean degree for the sizes below and
            above the critical_size.

        Returns
        -------

        deg_small_arr : np.ndarray of float of shape (n_steps,)
            The mean node degree at each timestep, for the sizes smaller
            than critical_size.

        deg_large_arr : np.ndarray of float of shape (n_steps,)
            The mean node degree at each timestep, for the sizes larger
            than critical_size.
        """
        n_steps = len(self.time_series)
        deg_small_arr = np.zeros(n_steps)
        deg_large_arr = np.zeros(n_steps)

        for i, net in enumerate(self.time_series):
            mean_deg_small, mean_deg_large = net.partial_mean_deg(
                critical_size)
            deg_small_arr[i] = mean_deg_small
            deg_large_arr[i] = mean_deg_large

        return deg_small_arr, deg_large_arr


    def compute_mean_str(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the mean graph strength for the time-series.

        Returns
        -------

        list_str : np.ndarray of float of shape (n_steps,)
            The mean node strength at each timestep. Notice that the
            mean in-strength and out-strength are the same, so only one
            number is necessary.

        list_in_std : np.ndarray of float of shape (n_steps,)
            The standard deviation of the in-strength distribution, at each
            timestep.

        list_out_std : np.ndarray of float of shape (n_steps,)
            The standard deviation of the out-strength distribution, at each
            timestep.
        """
        n_steps = len(self.time_series)
        list_str = np.zeros(n_steps)
        list_in_std = np.zeros(n_steps)
        list_out_std = np.zeros(n_steps)

        for i, net in enumerate(self.time_series):
            strength, in_str_std, out_str_std = net.mean_str()
            list_str[i] = strength
            list_in_std[i] = in_str_std
            list_out_std[i] = out_str_std

        return list_str, list_in_std, list_out_std


    def compute_mean_cc(
        self,
        critical_size: int,
        thr: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the mean clustering coefficient for the time-series.

        Parameters
        ----------

        critical_size : int
            Distinguish between the mean clustering coefficient for the sizes
            below and above the critical_size.

        thr : float, default = 0.0
            In computing the clustering coefficinet, only consider a link if
            its weigth is higher than thr.

        Returns
        -------

        list_cc : np.ndarray of float of shape (n_steps,)
            The mean clustering coefficient at each timestep (it's a number
            between 0 -tree like graph- and 1 -fully connected graph-).

        cc_small_arr : np.ndarray of float of shape (n_steps,)
            The mean clustering coefficient at each timestep for the sizes
            smaller than critical_size.

        cc_large_arr : np.ndarray of float of shape (n_steps,)
            The mean clustering coefficient at each timestep for the sizes
            larger than critical_size.
        """
        n_steps = len(self.time_series)
        list_cc = np.zeros(n_steps)
        cc_small_arr = np.zeros(n_steps)
        cc_large_arr = np.zeros(n_steps)

        for i, net in enumerate(self.time_series):
            list_cc[i], cc_small, cc_large = net.mean_cc(thr, critical_size)
            cc_small_arr[i] = cc_small
            cc_large_arr[i] = cc_large

        return list_cc, cc_small_arr, cc_large_arr


    def compute_distances(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the diameter and average distance between nodes.

        Returns
        -------

        list_diam : np.ndarray of int of shape (n_steps,)
            The graph diameter, at each timestep.

        list_mean_dist : np.ndarray of float of shape (n_steps,)
            The mean note-to-node distance in the graph, at each timestep.
        """
        n_steps = len(self.time_series)
        list_diam = np.zeros(n_steps, dtype=int)
        list_mean_dist = np.zeros(n_steps)

        for i, net in enumerate(self.time_series):
            diam, mean_dist = net.diameter_and_average_distance()
            list_diam[i] = diam
            list_mean_dist[i] = mean_dist

        return list_diam, list_mean_dist
