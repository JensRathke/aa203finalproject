#!/usr/bin/env python3

import numpy as np
import cvxpy as cvx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import controller_tools as ct

from scipy.linalg import solve_discrete_are
from quadcopter import *
from plotting import *

class PQcopter_controller_MPC():
    """ Controller for a planar quadcopter using MPC """
    def __init__(self, qcopter: QuadcopterPlanar, s_init):
        """
        Functionality
            Initialisation of a controller for a planar quadcopter using MPC

        Parameters
            qcopter: quadcopter to be controlled
            s_init: initial state of the quadcopter
        """
        self.qc = qcopter
        
        self.n = 6                                  # state dimension
        self.m = 2                                  # control dimension
        self.Q = np.diag(np.array([10., 10., 1., 1., 10., 1.]))   # state cost matrix
        self.R = 10 * np.eye(self.m)                     # control cost matrix
        self.s_init = s_init                        # initial state
        self.s_goal = np.array([0., self.qc.h, 0., 0., 0. , 0.])      # goal state
        self.T = 30 # s                             # simulation time
        self.dt = 0.1 # s                           # sampling time
        self.K = int(self.T / self.dt) + 1          # number of steps
        self.N = 3                                  # rollout steps
        self.rs = 5.0
        self.ru = 0.1
        self.rT = np.inf

    def linearize_penalize(self, f, s, u):
        A, B = jax.jacobian(f, (0, 1))(s, u)
        P_dare = solve_discrete_are(A, B, self.Q, self.R)
        return A, B, P_dare
    
    def mpc_rollout(self, x0: np.ndarray, A: np.ndarray, B: np.ndarray, P: np.ndarray, Q: np.ndarray, R: np.ndarray, N: int, rx: float, ru: float, rf: float):
        """Solve the MPC problem starting at state `x0`."""
        n, m = Q.shape[0], R.shape[0]
        x_cvx = cvx.Variable((N + 1, n))
        u_cvx = cvx.Variable((N, m))

        # PART (a): YOUR CODE BELOW ###############################################
        # INSTRUCTIONS: Construct and solve the MPC problem using CVXPY.
        costs = []
        constraints = []

        for k in range(N + 1):
            if k == 0:
                # initial condition
                print(x0)
                constraints.append(x_cvx[k] == x0)

            if k == N:
                # terminal cost
                costs.append(cvx.quad_form(x_cvx[k] - self.s_goal, P))

                # terminal contraints
                constraints.append(cvx.norm(x_cvx[k] - self.s_goal, 'inf') <= rf)

            if k <= N and k > 0:
                # dynamics constraint
                constraints.append(A @ x_cvx[k-1] + B @ u_cvx[k-1] == x_cvx[k])

            if k < N:
                # stage cost
                costs.append(cvx.quad_form(x_cvx[k] - self.s_goal, Q))
                costs.append(cvx.quad_form(u_cvx[k], R))

                # state contraints
                # constraints.append(cvx.norm(x_cvx[k] - x_cvx[k-1], 'inf') <= rx)

                # control contraints
                constraints.append(cvx.norm(u_cvx[k], 'inf') <= ru)

        cost = cvx.sum(costs)

        # END PART (a) ############################################################

        prob = cvx.Problem(cvx.Minimize(cost), constraints)
        prob.solve()
        x = x_cvx.value
        u = u_cvx.value
        status = prob.status

        return x, u, status

    
    def land(self):
        t_line = np.arange(0., self.T, 1)

        x = np.copy(self.s_init)
        x_mpc = np.zeros((self.T, self.N + 1, self.n))
        x_mpc[0, 0] = self.s_init
        u_mpc = np.zeros((self.T, self.N, self.m))

        P = np.eye(self.n)

        # Initialize continuous-time and discretized dynamics
        f = jax.jit(self.qc.dynamics_jnp)
        fd = jax.jit(lambda s, u, dt=self.dt: s + dt*f(s, u))

        for t in range(1, self.T):
            # A, B, _ = self.linearize_penalize(self.qc.dynamics_jnp, x_mpc[t-1, 0], u_mpc[t-1, 0])

            A, B = jax.vmap(ct.linearize, in_axes=(None, 0, 0))(f, x_mpc[t-1, :-1], u_mpc[t-1])
            A, B = np.array(A[0]), np.array(B[0])

            x_mpc[t], u_mpc[t], status = self.mpc_rollout(x, A, B, P, self.Q, self.R, self.N, self.rs, self.ru, self.rT)
            print(x_mpc[t], u_mpc[t], status)
            if status == 'infeasible':
                x_mpc = x_mpc[:t]
                u_mpc = u_mpc[:t]
                break
            x = A@x + B@u_mpc[t, 0, :]

        x_values = np.zeros((x_mpc.shape[0], self.N + 1))
        y_values = np.zeros((x_mpc.shape[0], self.N + 1))

        for index in range(x_mpc.shape[0]):
            x_values[index] = x_mpc[index, :, 0]
            y_values[index] = x_mpc[index, :, 1]

        print(x_mpc)

        self.qc.animate(t_line, x_mpc[:, 0], x_mpc[:, 0], "test_MPC")
        plot_trajectory("test_MPC_traj", x_values, y_values)

class PQC_controller_nlMPC():
    """ Controller for a planar quadcopter using non-linear MPC """
    def __init__(self, quadcopter: QuadcopterPlanar, s_init):
        """
        Functionality
            Initialisation of a controller for a planar quadcopter using non-linear MPC

        Parameters
            quadcopter: quadcopter to be controlled
            s_init: initial state of the quadcopter
        """
        self.n = 6                                  # state dimension
        self.m = 2                                  # control dimension
        self.Q = np.diag(np.array([1., 1., 1., 1., 1., 1.]))   # state cost matrix
        self.R = 10 * np.eye(self.m)                # control cost matrix
        self.s_init = s_init                        # initial state
        self.s_goal = np.array([0., self.qcopter.h, 0., 0., 0. , 0.])      # goal state
        self.T = 30  # s                            # simulation time
        self.dt = 0.1 # s                           # sampling time
        self.K = int(self.T / self.dt) + 1          # number of steps
        self.N = 3                                  # rollout steps

        self.qc = quadcopter
        self.dynamics = ct.RK4Integrator(self.qc.dynamics_jnp, self.dt)

        