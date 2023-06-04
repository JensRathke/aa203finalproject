#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

filepath = "Results/"

def write_textfile(filename, controller, params, P, Q, R, total_control_cost, touchdowntime, touchdownvels, total_time):
    try:
        with open(filepath + filename + ".md", 'w') as f:
            f.write(
                f"Controller: {controller}\n\n P / QN:\n {P}\n\n Q:\n {Q}\n\n R:\n {R}\n\n total control cost: {total_control_cost}\n\n time to touchdown: {touchdowntime} s\n\n touchdown velocities: {touchdownvels} m/s, m/s, rad/s\n\n time to simulate: {round(total_time, 2)} s\n\n Parameters: {params}"
                )
    except FileNotFoundError:
        print("The 'docs' directory does not exist")

    print("Saved results as:", filepath + filename + '.md')


if __name__ == "__main__":
    write_textfile("test", "c", np.diag(np.array([5., 5., 2., 40., 50., 10.])), np.diag(np.array([5., 5., 2., 40., 50., 10.])), np.diag(np.array([5., 5., 2., 40., 50., 10.])), 1, 2, 3, 4)