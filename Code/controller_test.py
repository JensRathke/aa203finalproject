#!/usr/bin/env python3

import numpy as np
import jax
import jax.numpy as jnp
import time
import cvxpy as cvx

from scipy.integrate import odeint
from scipy.optimize import minimize
from functools import partial
from tqdm import tqdm

from quadcopter import *

class PQcopter_controller_test():
    """ Test of controller for a planar quadcopter """
    def __init__(self, qcopter: QuadcopterPlanar, s_init):
        """
        Functionality
            Initialisation of a controller for a planar quadcopter using iLQR

        Parameters
            qcopter: quadcopter to be controlled
            s_init: initial state of the quadcopter
        """
        self.qcopter = qcopter
        self.s_init = s_init                        # initial state
        self.s_goal = np.array([0., self.qcopter.h, 0., 0., 0. , 0.])      # goal state
        self.T = 3.  # s                           # simulation time
        self.dt = 0.1 # s
        self.N = int(self.T / self.dt)

        self.g = 9.807         # gravity (m / s**2)
        self.m = 2.5           # mass (kg)
        self.l = 1.0           # half-length (m)
        self.I_zz = 1.0         # moment of inertia about the out-of-plane axis (kg * m**2)

    def land(self):
              
        cost = lambda z: self.dt * np.sum(np.square(z.reshape(int(self.N) + 1, 8)[:, -2:]))

        def constraints(z):
            states_and_controls = z.reshape(int(self.N) + 1, 8)
            states = states_and_controls[:, :6]
            controls = states_and_controls[:, -2:]
            constraint_list = [states[0] - self.s_init, states[-1] - self.s_goal]
            for i in range(int(self.N)):
                constraint_list.append(states[i + 1] - (states[i] + self.dt * self.qcopter.dynamics(states[i], controls[i])))
            return np.concatenate(constraint_list)


        z_guess = np.concatenate([np.linspace(self.s_init, self.s_goal, int(self.N) + 1), np.ones((int(self.N) + 1, 2))], -1).ravel()
        z_iterates = [z_guess]
        print("start")
        result = minimize(cost,
                        z_guess,
                        constraints={
                            'type': 'eq',
                            'fun': constraints
                        },
                        options={'maxiter': 1000},
                        callback=lambda z: z_iterates.append(z))
        print("end")
        z_iterates = np.stack(z_iterates)
        
        t = np.linspace(0, int(self.T), int(self.N) + 1)
        z = result.x.reshape(int(self.N) + 1, 8)

        sg = np.zeros((t.size, 6))

        self.qcopter.plot_states(t, z, "test_controller_s", ["x", "y", "dx", "dy", "theta", "omega"])
        # self.qcopter.plot_controls(t[0:self.N], u, "test_controller_u")
        self.qcopter.animate(t, z, sg, "test_controller")

if __name__ == "__main__":
    pass