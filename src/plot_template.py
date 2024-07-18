"""Do the plots reading the output of the main."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

T_CONV = 0.3

def func_0(x, A, tau):
    return A*(1 - np.exp(- x / tau))

def func_1(x, A, tau, C):
    return A*np.exp(- x / tau) + C

def func_2(x, A, tau, C):
    return A*x*np.exp(- x / tau) + C

def plot_deg(
    title: str,
    degrees: np.ndarray,
    strength: np.ndarray,
):
    """Plot the mean degree and strength."""
    time = np.linspace(0, degrees.size, degrees.size)*T_CONV

    fig, axes = plt.subplots(2, 1)

    axes[0].plot(time, degrees)
    axes[0].set_ylabel("Mean degree")
    axes[0].set_ylim(0.0, max(degrees)*1.05)

    popt, pcov = curve_fit(func_2, time, degrees, p0=[1, 10, 1])
    print(popt)
    y_fit = func_2(time, *popt)
    axes[0].plot(time, y_fit, c='black', ls='--', lw=1)

    axes[1].plot(time, strength)
    axes[1].set_ylabel("Mean strength")
    axes[1].set_ylim(0.0, max(strength)*1.05)

    popt, pcov = curve_fit(func_1, time, strength)
    print(popt)
    y_fit = func_1(time, *popt)
    axes[1].plot(time, y_fit, c='black', ls='--', lw=1)

    axes[1].set_xlabel("Time [ns]")
    fig.savefig(f"final_figures/{title}", dpi=600)

    plt.show()


def plot_cl_coeff(title: str, cl_coeff: np.ndarray):
    """Plot the mean clustering coefficient."""
    fig, axes = plt.subplots()
    time = np.linspace(0, cl_coeff.size, cl_coeff.size)*T_CONV
    axes.plot(time, cl_coeff)

    popt, pcov = curve_fit(func_2, time, cl_coeff, p0=[1, 10, 1])
    print(popt)
    y_fit = func_2(time, *popt)
    axes.plot(time, y_fit, c='black', ls='--', lw=1)

    axes.set_xlabel("Time [ns]")
    axes.set_ylabel("Mean clustering coefficient")
    fig.savefig(f"final_figures/{title}", dpi=600)
    plt.show()


def plot_distances(
    title: str,
    diam: np.ndarray,
    m_dist: np.ndarray,
):
    """Plot the mean degree and strength."""
    fig, axes = plt.subplots(2, 1)
    time = np.linspace(0, diam.size, diam.size)*T_CONV
    axes[0].plot(time, diam)
    axes[1].plot(time, m_dist)

    popt, pcov = curve_fit(func_0, time, diam)
    print(popt)
    y_fit = func_0(time, *popt)
    axes[0].plot(time, y_fit, c='black', ls='--', lw=1)

    popt, pcov = curve_fit(func_0, time, m_dist)
    print(popt)
    y_fit = func_0(time, *popt)
    axes[1].plot(time, y_fit, c='black', ls='--', lw=1)

    axes[1].set_xlabel("Time [ns]")
    axes[0].set_ylabel("Graph diameter")
    axes[1].set_ylabel("Mean node distance")
    fig.savefig(f"final_figures/{title}", dpi=600)
    plt.show()


deg, _, _ = np.load("output_data/degrees.npy")
strn, _, _ = np.load("output_data/strength.npy")
clust_coeff = np.load("output_data/clust_coeff.npy")
diameter, mean_dist = np.load("output_data/distances.npy")

plot_deg("Degrees.png", deg, strn)

plot_cl_coeff("Clust_coeff.png", clust_coeff)

plot_distances("Distances.png", diameter, mean_dist)
