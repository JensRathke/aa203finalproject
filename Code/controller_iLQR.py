#!/usr/bin/env python3

import numpy as np
import jax
import jax.numpy as jnp
import time
import cvxpy as cp
import controller_tools as ct

from scipy.integrate import odeint
from scipy.optimize import minimize
from functools import partial
from tqdm.auto import tqdm

from quadcopter import *

class QC_controller_iLQR():
    """ Controller for a planar quadcopter using iLQR """
    def __init__(self, quadcopter: QuadcopterPlanar, n, m, QN, Q, R, s_init, s_goal, T, dt):
        """
        Functionality
            Initialisation of a controller for a planar quadcopter using iLQR

        Parameters
            qcopter: quadcopter to be controlled
            s_init: initial state of the quadcopter
        """
        self.qc = quadcopter
        self.n = n                                  # state dimension
        self.m = m                                  # control dimension
        self.Q = Q                                  # state cost matrix
        self.R = R                                  # control cost matrix
        self.P = QN                                # terminal state cost matrix
        self.s_init = s_init                        # initial state
        self.s_goal = s_goal                        # goal state
        self.T = T #s                               # simulation time
        self.dt = dt #s                             # sampling time

        self.landed = False

        self.timeline = None
        self.pad_trajectory = None

        self.description = "iLQR"
        self.params = ""

    def ilqr(self, f, s_init, s_goal, N, Q, R, QN, eps = 1e-3, max_iters = 1000):
        if max_iters <= 1:
            raise ValueError('Argument `max_iters` must be at least 1.')
        n = Q.shape[0]        # state dimension
        m = R.shape[0]        # control dimension

        # Initialize gains `Y` and offsets `y` for the policy
        Y = np.zeros((N, m, n))
        y = np.zeros((N, m))

        # Initialize the nominal trajectory `(s_bar, u_bar`), and the
        # deviations `(ds, du)`
        u_bar = np.zeros((N, m))
        s_bar = np.zeros((N + 1, n))
        s_bar[0] = s_init
        for k in range(N):
            s_bar[k+1] = f(s_bar[k], u_bar[k])
        ds = np.zeros((N + 1, n))
        du = np.zeros((N, m))

        # iLQR loop
        converged = False
        for iteration in tqdm(range(max_iters)):
            # Linearize the dynamics at each step `k` of `(s_bar, u_bar)`
            A, B = jax.vmap(ct.linearize, in_axes=(None, 0, 0))(f, s_bar[:-1], u_bar)
            A, B = np.array(A), np.array(B)

            # PART (c) ############################################################
            # Backward pass
            P = QN
            p = QN @ (s_bar[-1] - s_goal)
            for k in range(N-1, -1, -1):
                q = Q @ (s_bar[k] - s_goal)
                r = R @ u_bar[k]
                Hss = Q + A[k].T @ P @ A[k]
                Huu = R + B[k].T @ P @ B[k]
                Hsu = A[k].T @ P @ B[k]
                hs = q + A[k].T @ p
                hu = r + B[k].T @ p
                Y[k] = -np.linalg.solve(Huu, Hsu.T)
                y[k] = -np.linalg.solve(Huu, hu)
                # P = Hss - Y[k].T @ Huu @ Y[k]
                # p = hs - Y[k].T @ Huu @ y[k]
                P = Hss + Hsu @ Y[k]
                p = hs + Hsu @ y[k]

            # Forward pass
            for k in range(N):
                du[k] = y[k] + Y[k] @ ds[k]
                ds[k+1] = f(s_bar[k] + ds[k], u_bar[k] + du[k]) - s_bar[k+1]

            s_bar += ds
            u_bar += du
            #######################################################################

            if np.max(np.abs(du)) < eps:
                converged = True
                break
        if not converged:
            print('iLQR did not converge!')
            #raise RuntimeError('iLQR did not converge!')
        return s_bar, u_bar, Y, y

    def land(self):
        total_control_cost = 0.0
        touchdownvels = np.zeros(3)
        touchdowntime = 0

        # Initialize continuous-time and discretized dynamics
        f = jax.jit(self.qc.dynamics)
        fd = jax.jit(lambda s, u, dt=self.dt: s + dt*f(s, u))

        # Compute the iLQR solution with the discretized dynamics
        print('Computing iLQR solution ... ', end='', flush=True)
        start = time.time()
        t = np.arange(0., self.T, self.dt)
        N = t.size
        s_bar, u_bar, Y, y = self.ilqr(fd, self.s_init, self.s_goal, N, self.Q, self.R, self.P)
        print('done! ({:.2f} s)'.format(time.time() - start), flush=True)

        # Simulate on the true continuous-time system
        print('Simulating ... ', end='', flush=True)
        start = time.time()
        s = np.zeros((N + 1, self.n))
        u = np.zeros((N + 1, self.m))
        s[0] = self.s_init

        for k in tqdm(range(N - 1)):
            u[k] = u_bar[k] + y[k] + Y[k] @ (s[k] - s_bar[k])
            s[k+1] = odeint(lambda s, t: f(s, u[k]), s[k], t[k:k+2])[1]

        print('done! ({:.2f} s)'.format(time.time() - start), flush=True)

        return s, u, total_control_cost, self.landed, touchdowntime, touchdownvels        

if __name__ == "__main__":
    pass