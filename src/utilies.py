"""Some basic plot functions for the results."""


def plot_1(
    degrees,
    in_degrees_std,
    out_degrees_std,
    strength,
    in_stren_std,
    out_stren_std
):
    """Plot the mean degree and strength."""
    time = range(degrees.size)

    fig0, ax0 = plt.subplots(2, 1)
    in_degree_plus = degrees + in_degrees_std
    in_degree_minus = degrees - in_degrees_std
    out_degree_plus = degrees + out_degrees_std
    out_degree_minus = degrees - out_degrees_std

    ax0[0].plot(time, degrees)
    ax0[0].fill_between(time, in_degree_minus, in_degree_plus,
        alpha=0.25)
    ax0[1].plot(time, degrees)
    ax0[1].fill_between(time, out_degree_minus, out_degree_plus,
        alpha=0.25)

    ax0[1].set_xlabel("Time")
    ax0[0].set_ylabel("Mean in-degree")
    ax0[1].set_ylabel("Mean out-degree")
    ax0[0].set_ylim(0.0, max(in_degree_plus)*1.05)
    ax0[1].set_ylim(0.0, max(out_degree_plus)*1.05)
    fig0.savefig("output_figures/Fig1.png", dpi=600)

    fig1, ax1 = plt.subplots(2, 1)
    in_strength_plus = strength + in_stren_std
    in_strength_minus = strength - in_stren_std
    out_strength_plus = strength + out_stren_std
    out_strength_minus = strength - out_stren_std

    ax1[0].plot(time, strength)
    ax1[0].fill_between(time, in_strength_minus, in_strength_plus,
        alpha=0.25)
    ax1[1].plot(time, strength)
    ax1[1].fill_between(time, out_strength_minus, out_strength_plus,
        alpha=0.25)

    ax1[1].set_xlabel("Time")
    ax1[0].set_ylabel("Mean in-strength")
    ax1[1].set_ylabel("Mean out-strength")
    ax1[0].set_ylim(0.0, max(in_strength_plus)*1.05)
    ax1[1].set_ylim(0.0, max(out_strength_plus)*1.05)
    fig1.savefig("output_figures/Fig2.png", dpi=600)

    plt.show()


def plot_2(clust_coeff):
    """Plot the mean clustering coefficient."""
    fig, ax0 = plt.subplots()
    ax0.plot(clust_coeff)
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Mean clustering coefficient")
    plt.show()
    fig.savefig("output_figures/Fig3.png", dpi=600)


def plot_3(
    diameter,
    mean_dist
):
    """Plot the mean degree and strength."""
    fig, ax0 = plt.subplots(2, 1)
    ax0[0].plot(diameter)
    ax0[1].plot(mean_dist)
    ax0[1].set_xlabel("Time")
    ax0[0].set_ylabel("Graph diameter")
    ax0[1].set_ylabel("Mean node distance")

    plt.show()
    fig.savefig("output_figures/Fig4.png", dpi=600)