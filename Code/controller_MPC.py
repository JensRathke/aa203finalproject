#!/usr/bin/env python3

import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import controller_tools as ct

from scipy.linalg import solve_discrete_are
from functools import partial
from time import time
from tqdm.auto import tqdm
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
        self.Q = jnp.diag(jnp.array([10., 10., 1., 1., 10., 1.]))   # state cost matrix
        self.R = 2 * jnp.eye(self.m)                     # control cost matrix
        self.s_init = s_init                        # initial state
        self.s_goal = jnp.array([0., self.qc.h, 0., 0., 0. , 0.])      # goal state
        self.T = 30 # s                             # simulation time
        self.dt = 0.1 # s                           # sampling time
        self.K = int(self.T / self.dt) + 1          # number of steps
        self.N = 3                                  # rollout steps
        self.rs = 5.0
        self.ru = 0.1
        self.rT = jnp.inf

    def linearize_penalize(self, f, s, u):
        A, B = jax.jacobian(f, (0, 1))(s, u)
        P_dare = solve_discrete_are(A, B, self.Q, self.R)
        return A, B, P_dare
    
    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def affinize(self, f, s, u):
        """Affinize the function `f(s, u)` around `(s, u)`."""
        A, B  = jax.jacobian(f, (0, 1))(s, u)
        c = f(s, u) - A @ s - B @ u
        return A, B, c
    
    def mpc_rollout(self, x0: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray, P: jnp.ndarray, Q: jnp.ndarray, R: jnp.ndarray, N: int, rx: float, ru: float, rf: float):
        """Solve the MPC problem starting at state `x0`."""
        n, m = Q.shape[0], R.shape[0]
        x_cvx = cp.Variable((N + 1, n))
        u_cvx = cp.Variable((N, m))

        # PART (a): YOUR CODE BELOW ###############################################
        # INSTRUCTIONS: Construct and solve the MPC problem using CVXPY.
        costs = []
        constraints = []

        for k in range(N + 1):
            if k == 0:
                # initial condition
                constraints.append(x_cvx[k] == x0)

            if k == N:
                # terminal cost
                costs.append(cp.quad_form(x_cvx[k] - self.s_goal, P))

                # terminal contraints
                # constraints.append(cp.norm(x_cvx[k] - self.s_goal, 'inf') <= rf)

            if k <= N and k > 0:
                # dynamics constraint
                constraints.append(A[k-1] @ x_cvx[k-1] + B[k-1] @ u_cvx[k-1] == x_cvx[k])

            if k < N:
                # stage cost
                costs.append(cp.quad_form(x_cvx[k] - self.s_goal, Q))
                costs.append(cp.quad_form(u_cvx[k], R))

                # state contraints
                # constraints.append(cvx.norm(x_cvx[k] - x_cvx[k-1], 'inf') <= rx)

                # control contraints
                # constraints.append(cp.norm(u_cvx[k], 'inf') <= ru)

        cost = cp.sum(costs)

        # END PART (a) ############################################################

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        x = x_cvx.value
        u = u_cvx.value
        status = prob.status

        return x, u, status

    
    def land(self):
        t_line = jnp.arange(0., self.T, 1)

        x = jnp.copy(self.s_init)
        x_mpc = np.zeros((self.T, self.N + 1, self.n))
        x_mpc[0, 0] = self.s_init
        u_mpc = np.zeros((self.T, self.N, self.m))

        P = jnp.eye(self.n)

        # Initialize continuous-time and discretized dynamics
        f = jax.jit(self.qc.dynamics)
        # fd = jax.jit(ct.discretize(self.qc.dynamics_jnp, self.dt))
        fd = self.qc.discrete_dynamics

        for t in range(1, self.T):
            # A, B, _ = self.linearize_penalize(self.qc.dynamics_jnp, x_mpc[t-1, 0], u_mpc[t-1, 0])
            # A, B = jax.vmap(ct.linearize, in_axes=(None, 0, 0))(f, x_mpc[t-1, :-1], u_mpc[t-1])

            # A, B = ct.linearize(self.qc.discrete_dynamics_jnp, x_mpc[t-1, 0], u_mpc[t-1, 0])

            A, B, c = ct.affinize(fd, x_mpc[t-1, :-1], u_mpc[t-1])
            A, B = np.array(A), np.array(B)

            print("x1:", x)
            x_mpc[t], u_mpc[t], status = self.mpc_rollout(x[0], A, B, P, self.Q, self.R, self.N, self.rs, self.ru, self.rT)
            print(x_mpc[t], u_mpc[t], status)
            if status == 'infeasible':
                x_mpc = x_mpc[:t]
                u_mpc = u_mpc[:t]
                break
            x = A @ x + B @ u_mpc[t, 0]

        x_values = jnp.zeros((x_mpc.shape[0], self.N + 1))
        y_values = jnp.zeros((x_mpc.shape[0], self.N + 1))

        for index in range(x_mpc.shape[0]):
            x_values[index] = x_mpc[index, :, 0]
            y_values[index] = x_mpc[index, :, 1]

        self.qc.animate(t_line, x_mpc[:, 0], x_mpc[:, 0], "test_MPC")
        plot_trajectory("test_MPC_traj", x_values, y_values)