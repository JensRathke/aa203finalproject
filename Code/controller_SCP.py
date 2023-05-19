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

class PQcopter_controller_SCP():
    """ Controller for a planar quadcopter using SCP"""
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
        self.Q = np.diag([1e-2,1e-2, 1.,  1., 1e-3, 1e-3])    # state cost matrix
        self.R = 1e-2*np.eye(self.m)                     # control cost matrix
        self.P = 1e3*np.eye(self.n)                     # terminal state cost matrix
        self.s_init = s_init                        # initial state
        self.s_goal = np.array([0., self.qcopter.h, 0., 0., 0. , 0.])      # goal state
        self.T = 300.                                # simulation time
        self.dt = 0.1 # s
        self.u_max = 80000.                           # control effort bound
        self.eps = 5e-1
        self.ρ = 1.
        self.max_iters = 200
       # @partial(jax.jit, static_argnums=(0,))
       # @partial(jax.vmap, in_axes=(None, 0, 0))
        f_dynamics = jax.vmap(self.qcopter.dynamics_jnp, in_axes=(0, None))
        f_dynamics = jax.vmap(f_dynamics, in_axes=(None, 0))
        #f = jax.jit(self.qcopter.dynamics)
        f = jax.jit(f_dynamics)
        f = self.qcopter.dynamics_jnp
        #self.fd = jax.jit(lambda s, u, dt=self.dt: s + dt*f(s, u))
        self.fd = jax.jit(self.discretize(f, self.dt))
        #self.fd = self.discretize(f, self.dt)
                              # sampling time

    #@partial(jax.jit, static_argnums=(0,))
    #@partial(jax.vmap, in_axes=(None, 0, 0))
    def affinize(self, f, s, u):

        A, B  = jax.jacobian(f,(0,1))(s,u)
        c = f(s,u) - A @ s - B @ u
        print("A", A)
        print("B", B)
        print("c", c)
        return A, B, c

    def discretize(self,f, dt):
        """Discretize continuous-time dynamics `f` via Runge-Kutta integration."""

        def integrator(s, u, dt=dt):
            k1 = dt * f(s, u)
            k2 = dt * f(s + k1 / 2, u)
            k3 = dt * f(s + k2 / 2, u)
            k4 = dt * f(s + k3, u)
            return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return integrator

    def scp_iteration(self, f, s0, s_goal, s_prev, u_prev, N, P, Q, R, u_max, ρ):
        """Solve a single SCP sub-problem for the cart-pole swing-up problem.

        Arguments
        ---------
        f : callable
            A function describing the discrete-time dynamics, such that
            `s[k+1] = f(s[k], u[k])`.
        s0 : numpy.ndarray
            The initial state (1-D).
        s_goal : numpy.ndarray
            The goal state (1-D).
        s_prev : numpy.ndarray
            The state trajectory around which the problem is convexified (2-D).
        u_prev : numpy.ndarray
            The control trajectory around which the problem is convexified (2-D).
        N : int
            The time horizon of the LQR cost function.
        P : numpy.ndarray
            The terminal state cost matrix (2-D).
        Q : numpy.ndarray
            The state stage cost matrix (2-D).
        R : numpy.ndarray
            The control stage cost matrix (2-D).
        u_max : float
            The bound defining the control set `[-u_max, u_max]`.
        ρ : float
            Trust region radius.

        Returns
        -------
        s : numpy.ndarray
            A 2-D array where `s[k]` is the open-loop state at time step `k`,
            for `k = 0, 1, ..., N-1`
        u : numpy.ndarray
            A 2-D array where `u[k]` is the open-loop state at time step `k`,
            for `k = 0, 1, ..., N-1`
        J : float
            The SCP sub-problem cost.
        """
        f_affine = jax.vmap(self.affinize, in_axes=(None, 0, 0))
        A, B, c = f_affine(f, s_prev[:-1], u_prev)
        A, B, c = np.array(A), np.array(B), np.array(c)
        n = Q.shape[0]
        m = R.shape[0]
        s_cvx = cvx.Variable((N + 1, n))
        u_cvx = cvx.Variable((N, m))

        # PART (c) ################################################################
        # INSTRUCTIONS: Construct the convex SCP sub-problem.
        objective = 0.
        constraints = []

        cost_terms = []

        #cost for terminal state
        cost_terms.append(cvx.quad_form((s_cvx[N]-s_goal), P))
        #initial constraint
        constraints.append(s_cvx[0]==s0)
        #terminal constraint
        constraints.append(cvx.norm_inf(s_cvx[N] - s_prev[N])<= ρ)

        for i in range(N):
            # append stage costs
            cost_terms.append(cvx.quad_form((s_cvx[i]-s_goal), Q))#np.transpose(s_cvx[i]-s_goal) @ Q @ (s_cvx[i]-s_goal)
            cost_terms.append(cvx.quad_form(u_cvx[i], R))

            # stage affine constraints and trust region/control constraints
            constraints.append(s_cvx[i+1] == A[i] @ s_cvx[i] + B[i] @ u_cvx[i] + c[i])
            constraints.append(cvx.norm_inf(s_cvx[i] - s_prev[i])<= ρ)
            constraints.append(cvx.norm_inf(u_cvx[i] - u_prev[i])<= ρ)
            constraints.append(cvx.abs(u_cvx[i])<= u_max)

        # END PART (c) ############################################################
        objective = cvx.sum(cost_terms)
        prob = cvx.Problem(cvx.Minimize(objective), constraints)
        prob.solve()

        if prob.status != 'optimal':
            raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)
        s = s_cvx.value
        u = u_cvx.value
        J = prob.objective.value
        return s, u, J

    def solve_landing_scp(self, f, s0, s_goal, N, P, Q, R, u_max, ρ, eps, max_iters):
        """Solve the quadcopter landing problem via SCP.

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
        P : numpy.ndarray
            The terminal state cost matrix (2-D).
        Q : numpy.ndarray
            The state stage cost matrix (2-D).
        R : numpy.ndarray
            The control stage cost matrix (2-D).
        u_max : float
            The bound defining the control set `[-u_max, u_max]`.
        ρ : float
            Trust region radius.
        eps : float
            Termination threshold for SCP.
        max_iters : int
            Maximum number of SCP iterations.

        Returns
        -------
        s : numpy.ndarray
            A 2-D array where `s[k]` is the open-loop state at time step `k`,
            for `k = 0, 1, ..., N-1`
        u : numpy.ndarray
            A 2-D array where `u[k]` is the open-loop state at time step `k`,
            for `k = 0, 1, ..., N-1`
        J : numpy.ndarray
            A 1-D array where `J[i]` is the SCP sub-problem cost after the i-th
            iteration, for `i = 0, 1, ..., (iteration when convergence occured)`
        """
        n = Q.shape[0]  # state dimension
        m = R.shape[0]  # control dimension

        # Initialize dynamically feasible nominal trajectories
        u = np.zeros((N, m))
        s = np.zeros((N + 1, n))
        s[0] = s0
        fd = self.fd
        for k in range(N):
            s[k+1] = fd(s[k], u[k])

        # Do SCP until convergence or maximum number of iterations is reached
        converged = False
        J = np.zeros(max_iters + 1)
        J[0] = np.inf
        for i in (prog_bar := tqdm(range(max_iters))):
            s, u, J[i + 1] = self.scp_iteration(f, s0, s_goal, s, u, N,
                                        P, Q, R, u_max, ρ)
            dJ = np.abs(J[i + 1] - J[i])
            prog_bar.set_postfix({'objective change': '{:.5f}'.format(dJ)})
            if dJ < eps:
                converged = True
                print('SCP converged after {} iterations.'.format(i))
                break
        if not converged:
            raise RuntimeError('SCP did not converge!')
        J = J[1:i+1]
        return s, u, J

    def land(self):
        # Initialize continuous-time and discretized dynamics
        #f = jax.jit(self.qcopter.dynamics)
        #fd = jax.jit(lambda s, u, dt=self.dt: s + dt*f(s, u))
        #fd = jax.jit(self.discretize(self.qcopter.dynamics, self.dt))
        fd = self.fd

        # Compute the SCP solution with the discretized dynamics
        print('Computing SCP solution ... ', end='', flush=True)
        start = time.time()
        t = np.arange(0., self.T, self.dt)
        N = t.size - 1
        s, u, J  = self.solve_landing_scp(fd, self.s_init, self.s_goal, N, self.P,self.Q, self.R,
                                                    self.u_max, self.ρ, self.eps, self.max_iters)
        print('done! ({:.2f} s)'.format(time.time() - start), flush=True)

        # Simulate on the true continuous-time system
        print('Simulating ... ', end='', flush=True)
        start = time.time()
        s = np.zeros((N + 1, self.n))
        u = np.zeros((N, self.m))
        s[0] = self.s_init

        for k in range(N):
            s[k+1] = fd(s[k], u[k])

        print('done! ({:.2f} s)'.format(time.time() - start), flush=True)

        sg = np.zeros((t.size, 6))
        sg[:, 1] = self.qcopter.h

        self.qcopter.plot_states(t, s, "test_SCP_s", ["x", "y", "dx", "dy", "theta", "omega"])
        self.qcopter.plot_controls(t[0:N], u, "test_SCP_u", ["T1", "T2"])
        self.qcopter.animate(t, s, sg, "test_SCP")


if __name__ == "__main__":
    pass