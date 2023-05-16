#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_1x2(filename, t, data00, data01, figure_title="figure", plot_titles=["", ""], xlabels=["time [s]", "time [s]"], ylabels=["", ""], fig_num = 1):
    """
    Functionality
        Creates 2 plots in 1 (stacked) by 2 (next to each other) shape
    
    Parameters
        filename: name and relative directory of the output file without file extention (e.g.: "dir/filename") 
        t: array of size N with the sequence of time stamps
        data00: array of size N with the data for plot 0, 0
        data01: array of size N with the data for plot 0, 1
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

    axs[0].plot(t, data00)
    axs[0].title.set_text(plot_titles[0])
    axs[0].set(xlabel=xlabels[0], ylabel=ylabels[0])

    axs[1].plot(t, data01)
    axs[1].title.set_text(plot_titles[1])
    axs[1].set(xlabel=xlabels[1], ylabel=ylabels[1])

    # Show figure
    fig.canvas.manager.set_window_title(figure_title)
    
    plt.savefig(filename + '.png')
    plt.show()
    plt.close(fig)

def plot_3x2(filename, t, data00, data01, data10, data11, data20, data21, figure_title="figure", plot_titles=["", "", "", "", "", ""], xlabels=["time [s]", "time [s]", "time [s]", "time [s]", "time [s]", "time [s]"], ylabels=["", "", "", "", "", ""], fig_num = 1):
    """
    Functionality
        Creates 6 plots in 3 (stacked) by 2 (next to each other) shape
    
    Parameters
        filename: name and relative directory of the output file without file extention (e.g.: "dir/filename") 
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

    # Show figure
    fig.canvas.manager.set_window_title(figure_title)
    
    plt.savefig(filename + '.png')
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    t = np.linspace(0, 10, 101)
    d00 = t
    d01 = -t
    d10 = np.sin(t)
    d11 = np.cos(t)
    d20 = -np.sin(t)
    d21 = -np.cos(t)

    plot_titles = ["title00", "title01", "title10", "title11", "title20", "title21"]
    xlabels = ["x00", "x01", "x10", "x11", "x20", "x21"]
    ylabels = ["y00", "y01", "y10", "y11", "y20", "y21"]

    # test plot_1x2
    plot_1x2('Figures/test_1x2', t, d00, d01, "1x2", ["title00", "title01"], ["x00", "x01"], ["y00", "y01"], 2)

    # test plot_3x2
    plot_3x2('Figures/test_3x2', t, d00, d01, d10, d11, d20, d21, "test figure title", plot_titles, xlabels, ylabels, 1)