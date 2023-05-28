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
from plotting import *
from animation import *

class SimulationPlanar:
    """ Simulation for a planar Quadcopter """
    def __init__(self, quadcopter: QuadcopterPlanar, controller, T, dt):
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
        k_buffer = 20                       # additional steps for the pad trajectory

        self.s_trajectory = np.zeros((self.K, self.controller.n))
        self.u_trajectory = np.zeros((self.K, self.controller.m))

        self.timeline = np.arange(0., self.K * self.dt, self.dt)
        self.timeline_pad = np.arange(0., (self.K + k_buffer) * self.dt, self.dt)
        self.pad_trajectory = np.zeros((self.K + k_buffer, self.controller.n)) # pad state [x, y, 0, 0, phi, 0]

        for k in range(self.K + k_buffer):
            self.pad_trajectory[k, 0] = 6 * np.sin(self.timeline_pad[k])
            self.pad_trajectory[k, 1] = 0
            self.pad_trajectory[k, 4] = 0.1 * np.pi * np.sin(self.timeline_pad[k])

        self.controller.pad_trajectory = self.pad_trajectory


    def simulate(self):
        """
        Functionality
            Runs the simulation
        """
        # Run the simulation
        total_time = time()
        self.s_trajectory, self.u_trajectory, total_control_cost = self.controller.land()
        total_time = time() - total_time

        # Print results
        print('Total elapsed time:', total_time, 'seconds')
        print('Total control cost:', total_control_cost)
        
        # Plot trajectory
        self.qc.plot_trajectory(self.timeline, self.s_trajectory, "test_nlMPC_trajectory")

        # Plot states
        self.qc.plot_states(self.timeline, self.s_trajectory, "test_nlMPC_states")
        
        # Plot states
        self.qc.plot_controls(self.timeline, self.u_trajectory, "test_nlMPC_controls")

        # Create animation
        self.qc.animate(self.timeline, self.s_trajectory, self.pad_trajectory, "test_nlMPC")


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