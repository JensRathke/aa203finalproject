#!/usr/bin/env python3

import numpy as np

from time import time

from simulation import *
from quadcopter import *
from spacecraft import *
from controller_iLQR import *
from controller_SCP import *
from controller_MPC import *
from controller_nlMPC import *
from plotting import *
from animation import *
from markdown import write_textfile

class SimulationPlanar:
    """ Simulation for a planar Quadcopter """
    def __init__(self, quadcopter: QuadcopterPlanar, controller, T, dt, k_buffer = 20, output_filename = "test_sim"):
        """
        Functionality
            Initialisation
        
        Parameters
            
        """
        self.qc = quadcopter
        self.controller = controller
        self.T = T #s                       # simulation time
        self.dt = dt #s                     # sampling time
        self.K = int(self.T / self.dt) + 1  # number of steps
        k_buffer = k_buffer                 # additional steps for the pad trajectory

        self.output_filename = output_filename

        self.s_trajectory = np.zeros((self.K, self.controller.n))
        self.u_trajectory = np.zeros((self.K, self.controller.m))

        self.timeline = np.arange(0., self.K * self.dt, self.dt)
        self.timeline_pad = np.arange(0., (self.K + k_buffer) * self.dt, self.dt)
        self.pad_trajectory = np.zeros((self.K + k_buffer, self.controller.n)) # pad state [x, y, 0, 0, phi, 0]

        for k in range(self.K + k_buffer):
            self.pad_trajectory[k, 0] = 6 * np.cos(self.timeline_pad[k] / 4)
            self.pad_trajectory[k, 1] = 0.1 * np.cos(self.timeline_pad[k] * 3 / 4)
            self.pad_trajectory[k, 4] = 0.06 * np.pi * np.sin(self.timeline_pad[k] * 2 / 4)

        self.controller.timeline = self.timeline
        self.controller.pad_trajectory = self.pad_trajectory


    def simulate(self):
        """
        Functionality
            Runs the simulation
        """
        # Run the simulation
        total_time = time()
        self.s_trajectory, self.u_trajectory, total_control_cost, landed, touchdowntime, touchdownvels = self.controller.land()
        total_time = time() - total_time

        # Print results
        print("total control cost:", round(total_control_cost, 2))
        print("time to touchdown: ", touchdowntime, "s")
        print("touchdown velocities: ", touchdownvels)
        print("time to simulate: ", round(total_time, 2), "s")

        # Write results
        self.write_results(total_control_cost, touchdowntime, touchdownvels, total_time)

        # Plot trajectory
        self.plot_trajectory(self.timeline, self.s_trajectory, self.output_filename + "_trajectory")

        # Plot states
        self.plot_states(self.timeline, self.s_trajectory, self.output_filename + "_states", ["x", "y", "dx", "dy", "phi", "omega"])
        
        # Plot controls
        self.plot_controls(self.timeline, self.u_trajectory, self.output_filename + "_controls", [r"$T_1$", r"$T_2$"])

        # Plot state costs
        self.plot_statecosts()

        # Create animation
        self.animate(self.timeline, self.s_trajectory, self.pad_trajectory, self.output_filename)

    def write_results(self, total_control_cost, touchdowntime, touchdownvels, total_time):
        write_textfile(self.output_filename + "_results", self.controller.description, self.controller.P, self.controller.Q, self.controller.R, total_control_cost, touchdowntime, touchdownvels, total_time)
    
    def plot_statecosts(self):
        costs = np.zeros((self.timeline.size, self.controller.n + 1))

        for statevarible in range(self.controller.n):
            costs[:, statevarible] = (self.s_trajectory[:, statevarible] - self.pad_trajectory[0 : self.timeline.size, statevarible]) ** 2 * self.controller.Q[statevarible, statevarible]

        for t in range(self.timeline.size):
            costs[t, -1] = (self.s_trajectory[t] - self.pad_trajectory[t]).T @ self.controller.Q @ (self.s_trajectory[t] - self.pad_trajectory[t])

        plot_1x1(self.output_filename + "_statecosts", self.timeline, costs.T, "State Costs", show_legend=True, legend_labels=["x", "y", "dx", "dy", "phi", "omega", "total cost"])

    def plot_states(self, t, s, filename, plot_titles=["", "", "", "", "", ""], y_labels=["", "", "", "", "", ""]):
        """
        Functionality
            Plot quadcopter states

        Parameters
            t: time
            s: state trajectory (x, y, dx, dy, psi, omega)
            filename: name of the output file without file-extension
        """
        plot_3x2(filename, t, s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5], "trajectory", plot_titles=plot_titles, ylabels=y_labels)

    def plot_controls(self, t, u, filename, plot_titles=["", ""], y_labels=["", ""]):
        """
        Functionality
            Plot a quadcopter trajectory

        Parameters
            t: time
            u: controls (t1, t2)
            filename: name of the output file without file-extension
        """
        plot_1x2(filename, t, u[:, 0], u[:, 1], "controls", plot_titles=plot_titles, ylabels=y_labels)

    def plot_trajectory(self, t, s, filename):
        """
        Functionality
            Plot a quadcopter trajectory

        Parameters
            t: time
            s: state trajectory (x, y, dx, dy, psi, omega)
            filename: name of the output file without file-extension
        """
        plot_trajectory(filename, s[:, 0], s[:, 1], "trajectory")

    def animate(self, t, s, sg, filename):
        """
        Functionality
            Animate a quadcopter trajectory

        Parameters
            t: time
            s: state trajectory (x, y, dx, dy, psi, omega)
            sg: goal state trajectory (x, y, dx, dy, psi, omega)
            filename: name of the output file without file-extension
        """
        animate_planar_quad(filename, t, s[:, 0], s[:, 1], s[:, 4], sg[:, 0], sg[:, 1], sg[:, 4], self.qc.l, self.qc.r, self.qc.h)


class SimulationCubic:
    """ Simulation for a cubic Quadcopter """
    def __init__(self, quadcopter: QuadcopterCubic):
        """
        Functionality
            Initialisation
        
        Parameters
            
        """

if __name__ == "__main__":
    pass