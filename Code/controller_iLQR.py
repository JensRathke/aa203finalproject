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

class PQcopter_controller_iLQR():
    """ Controller for a planar quadcopter using iLQR """
    def __init__(self, qcopter: QuadcopterPlanar, s_init):
        """
        Functionality
            Initialisation of a controller for a planar quadcopter using iLQR

        Parameters
            qcopter: quadcopter to be controlled
            s_init: initial state of the quadcopter
        """
        self.qcopter = qcopter

        self.n = 6                                  # state dimension
        self.m = 2                                  # control dimension
        self.Q = np.diag(np.array([2., 2., 1., 1., 2., 1.]))   # state cost matrix
        #self.Q = np.diag(jnp.array([30., 30., 1., 1., 30., 1.]))
        self.R = 2.*np.eye(self.m)                     # control cost matrix
        self.QN = 1e3*np.eye(self.n)                     # terminal state cost matrix
        self.s_init = s_init                        # initial state
        self.s_goal = np.array([0., self.qcopter.h, 0., 0., 0. , 0.])      # goal state
        self.T = 30.  # s                           # simulation time
        self.dt = 0.1 # s                           # sampling time

    def linearize(self, f, s, u):
        A, B = jax.jacobian(f, (0, 1))(s, u)
        return A, B

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
        for iteration in range(max_iters):
            # Linearize the dynamics at each step `k` of `(s_bar, u_bar)`
            #print("s_bar[:-1], u_bar", s_bar[:-1], u_bar)
            A, B = jax.vmap(self.linearize, in_axes=(None, 0, 0))(f, s_bar[:-1], u_bar)
            A, B = np.array(A), np.array(B)

            # PART (c) ############################################################
            # INSTRUCTIONS: Update `Y`, `y`, `ds`, `du`, `s_bar`, and `u_bar`.

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

    def ilqr_hk(self, f, s0, s_goal, N, Q, R, QN, eps=1e-3, max_iters=1000):
        """Compute the iLQR set-point tracking solution.

        Arguments
        ---------
        f : callable
            A function describing the discrete-time dynamics, such that
            `s[k+1] = f(s[k], u[k])`.
        s0 : numpy.ndarray
            The initial state (1-D).
        s_goal : numpy.ndarray
            The goal state (1-D).
        N : int
            The time horizon of the LQR cost function.
        Q : numpy.ndarray
            The state cost matrix (2-D).
        R : numpy.ndarray
            The control cost matrix (2-D).
        QN : numpy.ndarray
            The terminal state cost matrix (2-D).
        eps : float, optional
            Termination threshold for iLQR.
        max_iters : int, optional
            Maximum number of iLQR iterations.

        Returns
        -------
        s_bar : numpy.ndarray
            A 2-D array where `s_bar[k]` is the nominal state at time step `k`,
            for `k = 0, 1, ..., N-1`
        u_bar : numpy.ndarray
            A 2-D array where `u_bar[k]` is the nominal control at time step `k`,
            for `k = 0, 1, ..., N-1`
        Y : numpy.ndarray
            A 3-D array where `Y[k]` is the matrix gain term of the iLQR control
            law at time step `k`, for `k = 0, 1, ..., N-1`
        y : numpy.ndarray
            A 2-D array where `y[k]` is the offset term of the iLQR control law
            at time step `k`, for `k = 0, 1, ..., N-1`
        """
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
        s_bar[0] = s0
        for k in range(N):
            s_bar[k+1] = f(s_bar[k], u_bar[k])
        ds = np.zeros((N + 1, n))
        du = np.zeros((N, m))

        # iLQR loop
        converged = False
        for i in range(max_iters):
            # Linearize the dynamics at each step `k` of `(s_bar, u_bar)`
            A, B = jax.vmap(self.linearize, in_axes=(None, 0, 0))(f, s_bar[:-1], u_bar)
            A, B = np.array(A), np.array(B)

            # PART (c) ############################################################
            # INSTRUCTIONS: Update `Y`, `y`, `ds`, `du`, `s_bar`, and `u_bar`.
            P_T = QN
            P_k = P_T
            p_T= q_T = QN.T@ (s_bar[-1]-s_goal)
            p_k=p_T
            q_k=q_T
            r_T = R.T@ u_bar[-1]
            r_k= r_T
            #backward recursion
            for j in range(N-1, -1, -1):
                q_k = Q.T @ (s_bar[j]-s_goal)
                r_k = R.T @ u_bar[j]
                hxt = q_k + A[j].T @ p_k
                hut = r_k + B[j].T @ p_k
                Hxxt = Q + A[j].T @ P_k @ A[j]
                Huut = R + B[j].T @ P_k @ B[j]
                Hxut = A[j].T @ P_k @ B[j]

                #K_k= -1.0*LA.pinv(Huut)@Hxut.transpose()
                Y[j]= -1.0 * np.linalg.pinv(Huut) @ Hxut.T
                #k_k = -1.0 * LA.pinv(Huut)@hut
                y[j] = -1.0 * np.linalg.pinv(Huut) @ hut
                #p_k = hxt + Hxxt@k_k
                p_k = hxt + Hxut @ y[j]
                #P_k = Hxxt +Hxut@K_k
                P_k = Hxxt + Hxut @ Y[j]
                """
                if np.isnan(P_k).any():
                    import ipdb; ipdb.set_trace()
                """

            #forward pass
            #roll out
            for k in range(N):
                du[k] = y[k] + Y[k] @ ds[k]
                ds[k+1] = f(s_bar[k] + ds[k], u_bar[k] + du[k]) - s_bar[k+1]
                #s_bar[k+1] = s_bar[k+1] + ds[k+1]
                #u_bar[k] = u_bar[k] + du[k]
            s_bar+=ds
            u_bar+=du
            #######################################################################
            #print(np.max(np.abs(du)))
            if np.max(np.abs(du)) != 0.0 and np.max(np.abs(du)) < eps:
                converged = True
                print(i, np.max(np.abs(du)))
                break
        if not converged:
            # raise RuntimeError('iLQR did not converge!')
            print(s_bar, u_bar)
            print('j',j)
            print('i',i)
        return s_bar, u_bar, Y, y

    def land(self):
        # Initialize continuous-time and discretized dynamics
        f = jax.jit(self.qcopter.dynamics_jnp)
        fd = jax.jit(lambda s, u, dt=self.dt: s + dt*f(s, u))

        # Compute the iLQR solution with the discretized dynamics
        print('Computing iLQR solution ... ', end='', flush=True)
        start = time.time()
        t = np.arange(0., self.T, self.dt)
        N = t.size - 1
        s_bar, u_bar, Y, y = self.ilqr(fd, self.s_init, self.s_goal, N, self.Q, self.R, self.QN)
        print('done! ({:.2f} s)'.format(time.time() - start), flush=True)

        # Simulate on the true continuous-time system
        print('Simulating ... ', end='', flush=True)
        start = time.time()
        s = np.zeros((N + 1, self.n))
        u = np.zeros((N, self.m))
        s[0] = self.s_init

        for k in range(N):
            u[k] = u_bar[k] + y[k] + Y[k] @ (s[k] - s_bar[k])
            s[k+1] = odeint(lambda s, t: f(s, u[k]), s[k], t[k:k+2])[1]

        print('done! ({:.2f} s)'.format(time.time() - start), flush=True)

        sg = np.zeros((t.size, 6))

        self.qcopter.plot_states(t, s, "test_iLQR_s", ["x", "y", "dx", "dy", "theta", "omega"])
        self.qcopter.plot_controls(t[0:N], u, "test_iLQR_u", ["T1", "T2"])
        self.qcopter.animate(t, s, sg, "test_iLQR")

if __name__ == "__main__":
    pass