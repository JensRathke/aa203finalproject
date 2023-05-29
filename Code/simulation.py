#!/usr/bin/env python3

import numpy as np

from time import time

from simulation import *
from quadcopter import *
from spacecraft import *
from controller_test import *
from controller_iLQR import *
from controller_SCP import *
from controller_MPC import *
from controller_nlMPC import *
from plotting import *
from animation import *

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
            self.pad_trajectory[k, 0] = 6 * np.sin(self.timeline_pad[k])
            self.pad_trajectory[k, 1] = 0.1 * np.cos(self.timeline_pad[k] * 3)
            self.pad_trajectory[k, 4] = 0.06 * np.pi * np.sin(self.timeline_pad[k] * 2)

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
        total_time = touchdowntime - total_time

        # Print results
        print('total control cost:', round(total_control_cost, 2))
        print("time to touchdown: ", round(total_time, 2), "s")
        print("touchdown velocities: ", touchdownvels)

        # Plot trajectory
        self.qc.plot_trajectory(self.timeline, self.s_trajectory, self.output_filename + "_trajectory")

        # Plot states
        self.qc.plot_states(self.timeline, self.s_trajectory, self.output_filename + "_states", ["x", "y", "dx", "dy", "phi", "omega"])
        
        # Plot states
        self.qc.plot_controls(self.timeline, self.u_trajectory, self.output_filename + "_controls", [r"T_1", r"T_2"])

        # Create animation
        self.qc.animate(self.timeline, self.s_trajectory, self.pad_trajectory, self.output_filename)


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