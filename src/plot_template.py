"""Plots observables describing the graph along the time."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

T_CONV = 0.3


def func_n_nodes(x, A, tau):
    """Goes monotonically from 0 to A."""
    return A*(1 - np.exp(- x / tau))


def func_str(x, A, tau, C):
    """Goes monotonically from A + C to C."""
    return A*np.exp(- x / tau) + C


def func_deg(x, A, tau, C):
    """Goes from 0 to C with a maximum in x = tau + C/A."""
    return (A*x - C)*np.exp(- x / tau) + C


def func_cc(x, A, t1, t2, B):
    """Goes from B to 0 with a sharp initial peak."""
    return A * (1 - np.exp(-x / t1)) * np.exp(-x / t2) + B


def func_dist(x, tau, C):
    """Goes from B + C to C with a maximum in x = tau - B/A."""
    return (1 - C)*np.exp(- x / tau) + C


def plot_n_nodes(
    title: str,
    time_steps: np.ndarray,
    n_nodes: np.ndarray,
):
    """Plot the number of nodes."""
    time = time_steps*T_CONV

    fig, axes = plt.subplots()

    ### Number of nodes ###
    axes.plot(time, n_nodes)
    axes.set_ylabel(r"Number of nodes $n$")
    popt, pcov = curve_fit(func_n_nodes, time, n_nodes)
    y_fit = func_n_nodes(time, *popt)
    axes.plot(time, y_fit, c='red', ls='--', lw=1)

    print("\n### Number of nodes ###")
    print(f"Final value of {popt[0]} ("
        f"{np.sqrt(pcov[0][0])})")
    print(f"Typical timescale of {popt[1]} ({np.sqrt(pcov[1][1])})")
    print("#######################")

    axes.set_xscale('log')
    axes.set_xlabel("Time [ns]")

    plt.show()
    fig.savefig(f"final_figures/{title}", dpi=600)


def plot_deg(
    title: str,
    time_steps: np.ndarray,
    degrees: np.ndarray,
    strength: np.ndarray,
):
    """Plot the mean degree and strength."""
    time = time_steps*T_CONV

    fig, axes = plt.subplots(2, 1)

    ### Average node's degree ###
    axes[0].plot(time, degrees)
    axes[0].set_ylabel(r"Average degree $\langle k\rangle$")
    popt, pcov = curve_fit(func_deg, time, degrees, p0=[0.3, 50, 2])
    y_fit = func_deg(time, *popt)
    axes[0].plot(time, y_fit, c='red', ls='--', lw=1)

    print("\n### Average degree ###")
    print(f"Final value of {popt[2]} ("
        f"{np.sqrt(pcov[2][2])})")
    print(f"Typical timescale of {popt[1]} ({np.sqrt(pcov[1][1])})")
    print("#######################")

    ### Average node's strength ###
    axes[1].plot(time, strength)
    axes[1].set_ylabel(r"Average strength $\langle s\rangle$")
    popt, pcov = curve_fit(func_str, time, strength)
    y_fit = func_str(time, *popt)
    axes[1].plot(time, y_fit, c='red', ls='--', lw=1)

    print("\n### Average strength ###")
    print(f"Final value of {popt[2]} ("
        f"{np.sqrt(pcov[2][2])})")
    print(f"Typical timescale of {popt[1]} ({np.sqrt(pcov[1][1])})")
    print("#######################")

    for i in range(2):
        axes[i].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel("Time [ns]")

    plt.show()
    fig.savefig(f"final_figures/{title}", dpi=600)


def plot_partial_deg(
    title: str,
    time_steps: np.ndarray,
    deg_small: np.ndarray,
    deg_large: np.ndarray,
):
    """Plot the partial degrees."""
    time = time_steps*T_CONV

    fig, axes = plt.subplots()

    axes.plot(time, deg_small, label="Small")
    axes.plot(time, deg_large, label="Large")

    popt, pcov = curve_fit(func_deg, time, deg_small, p0=[0.7, 45, 4])
    y_fit = func_deg(time, *popt)
    axes.plot(time, y_fit, c='black', ls='--', lw=1)
    print("\n### Degree small ###")
    print(f"Final value of {popt[2]} ("
        f"{np.sqrt(pcov[2][2])})")
    print(f"Typical timescale of {popt[1]} ({np.sqrt(pcov[1][1])})")

    popt, pcov = curve_fit(func_deg, time, deg_large, p0=[0.5, 40, 2])
    y_fit = func_deg(time, *popt)
    axes.plot(time, y_fit, c='black', ls='--', lw=1)
    print("\n### Degree large ###")
    print(f"Final value of {popt[2]} ("
        f"{np.sqrt(pcov[2][2])})")
    print(f"Typical timescale of {popt[1]} ({np.sqrt(pcov[1][1])})")
    print("#######################")

    fig.suptitle("- The graph is considered undirected -")
    axes.set_ylabel(r"Average degree $\langle k\rangle$")
    axes.set_xscale('log')
    axes.set_xlabel("Time [ns]")
    axes.legend()

    plt.show()
    fig.savefig(f"final_figures/{title}", dpi=600)


def plot_cl_coeff(
    title: str,
    time_steps: np.ndarray,
    cl_coeff: np.ndarray
):
    """Plot the mean clustering coefficient."""
    fig, axes = plt.subplots()
    time = time_steps*T_CONV
    axes.plot(time, cl_coeff)

    popt, pcov = curve_fit(func_cc, time - time[0], cl_coeff,
        p0=[0.8, 0.1, 100, 0.06])
    y_fit = func_cc(time - time[0], *popt)
    axes.plot(time, y_fit, c='black', ls='--', lw=1)

    print("\n### Average clustering coeff ###")
    print(popt)
    print(f"Final value of {popt[3]} ("
        f"{np.sqrt(pcov[3][3])})")
    print(f"Typical timescale of {popt[2]} ({np.sqrt(pcov[2][2])})")
    print("################################")

    axes.set_xlabel("Time [ns]")
    axes.set_ylabel(r"Average clustering coefficient $\langle c\rangle$")
    axes.set_xscale('log')
    fig.savefig(f"final_figures/{title}", dpi=600)
    plt.show()


def plot_partial_cl_coeff(
    title: str,
    time_steps: np.ndarray,
    cl_coeff_small: np.ndarray,
    cl_coeff_large: np.ndarray,
):
    """Plot the mean clustering coefficient."""
    time = time_steps*T_CONV

    fig, axes = plt.subplots()

    axes.plot(time, cl_coeff_small, label='Small')
    axes.plot(time, cl_coeff_large, label='Large')

    print("\n### Partial clustering coeff ###")
    print("Small:")

    popt, pcov = curve_fit(func_cc, time - time[0], cl_coeff_small,
        p0=[0.8, 0.14, 120, 0.12])
    y_fit = func_cc(time - time[0], *popt)
    axes.plot(time, y_fit, c='black', ls='--', lw=1)

    print(f"Final value of {popt[3]} ("
        f"{np.sqrt(pcov[3][3])})")
    print(f"Typical timescale of {popt[2]} ({np.sqrt(pcov[2][2])})")

    popt, pcov = curve_fit(func_cc, time - time[0], cl_coeff_large,
        p0=[0.8, 1.0, 100, 0.06])
    y_fit = func_cc(time - time[0], *popt)
    axes.plot(time, y_fit, c='black', ls='--', lw=1)

    print("Large:")
    print(f"Final value of {popt[3]} ("
        f"{np.sqrt(pcov[3][3])})")
    print(f"Typical timescale of {popt[2]} ({np.sqrt(pcov[2][2])})")

    print("################################")

    axes.set_xlabel("Time [ns]")
    axes.set_ylabel(r"Average clustering coefficient $\langle c\rangle$")
    axes.set_xscale('log')
    axes.legend()

    fig.savefig(f"final_figures/{title}", dpi=600)
    plt.show()


def plot_distances(
    title: str,
    time_steps: np.ndarray,
    diam: np.ndarray,
    m_dist: np.ndarray,
):
    """Plot the mean degree and strength."""
    fig, axes = plt.subplots(2, 1)
    time = time_steps*T_CONV
    axes[0].plot(time, diam)
    axes[1].plot(time, m_dist)

    popt, pcov = curve_fit(func_dist, time, diam, p0=[104, 8])
    y_fit = func_dist(time, *popt)
    axes[0].plot(time, y_fit, c='red', ls='--', lw=1)

    print("\n### Diameter ###")
    print(f"Final value of {popt[1]} ("
        f"{np.sqrt(pcov[1][1])})")
    print(f"Typical timescale of {popt[0]} ({np.sqrt(pcov[0][0])})")
    print("#######################")

    popt, pcov = curve_fit(func_dist, time, m_dist, p0=[104, 1.3])
    y_fit = func_dist(time, *popt)
    axes[1].plot(time, y_fit, c='red', ls='--', lw=1)

    print("\n### Average node distance ###")
    print(f"Final value of {popt[1]} ("
        f"{np.sqrt(pcov[1][1])})")
    print(f"Typical timescale of {popt[0]} ({np.sqrt(pcov[0][0])})")
    print("################################")

    axes[1].set_xlabel("Time [ns]")
    axes[0].set_ylabel(r"Graph diameter $D$")
    axes[1].set_ylabel(r"Average node distance $\langle d\rangle$")
    axes[0].set_xscale('log')
    axes[1].set_xscale('log')
    fig.savefig(f"final_figures/{title}", dpi=600)
    plt.show()


time_steps = np.loadtxt("output_data/timesteps.txt")
n_nodes = np.loadtxt("output_data/n_nodes.txt")
deg, _, _ = np.load("output_data/degrees.npy")
strn, _, _ = np.load("output_data/strength.npy")
deg_small, deg_large = np.load("output_data/partial_degrees.npy")
clust_coeff = np.load("output_data/clust_coeff.npy")
cc_small, cc_large = np.load("output_data/partial_cc.npy")
diameter, mean_dist = np.load("output_data/distances.npy")

# plot_n_nodes("Nodes.png", time_steps, n_nodes)

# plot_deg("Degrees.png", time_steps, deg, strn)

# plot_partial_deg("Partials_deg.png", time_steps, deg_small, deg_large)

plot_cl_coeff("Clust_coeff.png", time_steps, clust_coeff)

plot_partial_cl_coeff("Partials_cc.png", time_steps, cc_small, cc_large)

# plot_distances("Distances.png", time_steps, diameter, mean_dist)
