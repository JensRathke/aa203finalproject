#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

filepath = "Figures/"

def plot_1x1(filename, t, data, figure_title="figure", plot_title="", xlabel="time [s]", ylabel="", show_legend=False, legend_labels=None, set_y_log=False, fig_num = 1):
    """
    Functionality
        Creates a plot
    
    Parameters
        filename: name and relative directory of the output file without file extention
        t: array of size N with the sequence of time stamps
        data: array of size N or n x N with the data to plot
        figure_title: title of the complete figure
        plot_title: array with titles of the individual plots
        xlabel: array with labels for the individual plots
        ylabel: array with labels for the individual plots
        fig_num: number of the figure
    """
    # Create legend labels if not provided
    if legend_labels == None:
        legend_labels = np.full(data.shape[0], "")

    # Create figure
    fig, axs = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True, num=fig_num)

    # Plot data
    fig.suptitle(figure_title)

    if data.ndim == 1:
        axs.plot(t, data, label=legend_labels)
    else:
        for i in range(data.shape[0]):
            axs.plot(t, data[i], label=legend_labels[i])

    if show_legend == True: axs.legend(loc='lower right')
    if set_y_log == True: axs.set_yscale('log')
    axs.title.set_text(plot_title)
    axs.set(xlabel=xlabel, ylabel=ylabel)

    fig.canvas.manager.set_window_title(figure_title)
    
    # Save figure
    plt.savefig(filepath + filename + '.png')
    print("Saved figure as:", filepath + filename + '.png')
    plt.show()
    plt.close(fig)

def plot_1x2(filename, t, data00, data01, figure_title="figure", plot_titles=["", ""], xlabels=["time [s]", "time [s]"], ylabels=["", ""], fig_num = 1):
    """
    Functionality
        Creates 2 plots in 1 (stacked) by 2 (next to each other) shape
    
    Parameters
        filename: name and relative directory of the output file without file extention
        t: array of size N with the sequence of time stamps
        data00: array of size N or n x N with the data for plot 0, 0
        data01: array of size N or n x N with the data for plot 0, 1
        figure_title: title of the complete figure
        plot_titles: array with titles of the individual plots
        xlabels: array with labels for the individual plots
        ylabels: array with labels for the individual plots
        fig_num: number of the figure
    """
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, num=fig_num)

    # Plot data
    fig.suptitle(figure_title)

    if data00.ndim == 1:
        axs[0].plot(t, data00)
    else:
        for i in range(data00.shape[0]):
            axs[0].plot(t, data00[i])

    axs[0].title.set_text(plot_titles[0])
    axs[0].set(xlabel=xlabels[0], ylabel=ylabels[0])

    if data01.ndim == 1:
        axs[1].plot(t, data01)
    else:
        for i in range(data01.shape[0]):
            axs[1].plot(t, data01[i])
    axs[1].title.set_text(plot_titles[1])
    axs[1].set(xlabel=xlabels[1], ylabel=ylabels[1])

    fig.canvas.manager.set_window_title(figure_title)
    
    # Save figure
    plt.savefig(filepath + filename + '.png')
    print("Saved figure as:", filepath + filename + '.png')
    plt.show()
    plt.close(fig)

def plot_3x2(filename, t, data00, data01, data10, data11, data20, data21, figure_title="figure", plot_titles=["", "", "", "", "", ""], xlabels=["time [s]", "time [s]", "time [s]", "time [s]", "time [s]", "time [s]"], ylabels=["", "", "", "", "", ""], fig_num = 1):
    """
    Functionality
        Creates 6 plots in 3 (stacked) by 2 (next to each other) shape
    
    Parameters
        filename: name and relative directory of the output file without file extention
        t: array of size N with the sequence of time stamps
        data00: array of size N with the data for plot 0, 0
        ...
        data21: array of size N with the data for plot 2, 1
        figure_title: title of the complete figure
        plot_titles: array with titles of the individual plots
        xlabels: array with labels for the individual plots
        ylabels: array with labels for the individual plots
        fig_num: number of the figure
    """
    # Create figure
    fig, axs = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True, num=fig_num) # gridspec_kw=dict(height_ratios=[1, 1, 1], width_ratios=[1, 1]), 

    # Plot data
    fig.suptitle(figure_title)

    axs[0, 0].plot(t, data00)
    axs[0, 0].title.set_text(plot_titles[0])
    axs[0, 0].set(xlabel=xlabels[0], ylabel=ylabels[0])

    axs[0, 1].plot(t, data01)
    axs[0, 1].title.set_text(plot_titles[1])
    axs[0, 1].set(xlabel=xlabels[1], ylabel=ylabels[1])

    axs[1, 0].plot(t, data10)
    axs[1, 0].title.set_text(plot_titles[2])
    axs[1, 0].set(xlabel=xlabels[2], ylabel=ylabels[2])

    axs[1, 1].plot(t, data11)
    axs[1, 1].title.set_text(plot_titles[3])
    axs[1, 1].set(xlabel=xlabels[3], ylabel=ylabels[3])

    axs[2, 0].plot(t, data20)
    axs[2, 0].title.set_text(plot_titles[4])
    axs[2, 0].set(xlabel=xlabels[4], ylabel=ylabels[4])

    axs[2, 1].plot(t, data21)
    axs[2, 1].title.set_text(plot_titles[5])
    axs[2, 1].set(xlabel=xlabels[5], ylabel=ylabels[5])

    fig.canvas.manager.set_window_title(figure_title)
    
    # Save figure
    plt.savefig(filepath + filename + '.png')
    print("Saved figure as:", filepath + filename + '.png')
    plt.show()
    plt.close(fig)

def plot_trajectory(filename, x, y, figure_title="figure", xlabel="x [m]", ylabel="y [m]", fig_num = 1):
    """
    Functionality
        Plots a 2D trajectory
    
    Parameters
        filename: name and relative directory of the output file without file extention
        x: array of size N with the data of x
        y: array of size N with the data of y
        figure_title: title of the complete figure
        xlabel: label for the x-axis
        ylabel: label for the y-axis
        fig_num: number of the figure
    """
    # Create figure
    fig, axs = plt.subplots(1, 1, constrained_layout=True, num=fig_num)

    # Plot data
    fig.suptitle(figure_title)

    if x.ndim == 1 and y.ndim == 1:
        axs.plot(x, y)
    elif x.ndim == y.ndim:
        for i in range(x.shape[0]):
            axs.plot(x[i], y[i])
    else:
        print("Error: x and y dims must be equal.")

    axs.set(xlabel=xlabel, ylabel=ylabel)

    fig.canvas.manager.set_window_title(figure_title)
    
    # Show figure
    plt.savefig(filepath + filename + '.png')
    print("Saved figure as:", filepath + filename + '.png')
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    t = np.linspace(0, 10, 101)
    d00 = np.tan(t)
    d01 = -np.tan(t)
    d10 = np.sin(t)
    d11 = np.cos(t)
    d20 = -np.sin(t)
    d21 = -np.cos(t)

    plot_titles = ["title00", "title01", "title10", "title11", "title20", "title21"]
    xlabels = ["x00", "x01", "x10", "x11", "x20", "x21"]
    ylabels = ["y00", "y01", "y10", "y11", "y20", "y21"]

    # test plot_1x1
    plot_1x1('test_1x1', t, np.array((d00, d10, d11)), "1x2", "title00", "x00", "y00", False, ["1", "2", "3"], False, 1)
    
    # test plot_1x2
    plot_1x2('test_1x2', t, np.array((d00, d10, d11)), np.array((d01, d20, d21)), "1x2", ["title00", "title01"], ["x00", "x01"], ["y00", "y01"], 2)

    # test plot_3x2
    plot_3x2('test_3x2', t, d00, d01, d10, d11, d20, d21, "test figure title", plot_titles, xlabels, ylabels, 3)

    # test plot trajectory
    plot_trajectory('test_traj', np.array((d10, d11)), np.array((d00, d10)), "test figure title", "x00", "y00", 4)