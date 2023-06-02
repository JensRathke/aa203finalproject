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


class QC_controller_SCP():
    """ Controller for a quadcopter using non-linear MPC """
    def __init__(self):
        """
        Functionality
            Initialisation of a controller for a quadcopter using non-linear MPC
        """
        raise NotImplementedError("Method must be overriden by a subclass of QC_controller_nlMPC")

    def landing_scp(self, s0, k0, s_init = None, u_init = None, convergence_error = False):
        # Initialize trajectory
        if s_init is None or u_init is None:
            s = np.zeros((self.N + 1, self.n))
            u = np.zeros((self.N, self.m))
            s[0] = s0
            for k in range(self.N):
                s[k+1] = self.dynamics(s[k], u[k])
        else:
            s[0] = np.copy(s_init)
            u[0] = np.copy(u_init)

        converged = False
        J = np.zeros(self.N_scp + 1)

        for iteration in range(self.N_scp):
            s, u, J[iteration + 1] = self.scp_iteration(s0, k0, s, u)

            dJ = np.abs(J[iteration + 1] - J[iteration])
            if dJ < self.eps:
                converged = True
                break

        if not converged and convergence_error:
            raise RuntimeError('SCP did not converge!')

        return s, u

    def scp_iteration(self, s0, k0, s_prev, u_prev):
        raise NotImplementedError("Method must be overriden by a subclass of QC_controller_SCP")

    def land(self):
        s = np.zeros((self.N + 1, self.n))
        u_tmp = np.zeros((self.N + 1 ,self.m))

        #s = np.copy(self.s_init)

        total_control_cost = 0.0
        tol = 0.1
        touchdownvels = np.zeros(3)
        touchdowntime = 0

        #s_init = None
        s_init = np.array([4., 45., 0., 0., -0.1 * np.pi, -1.])
        u_init = None

        #for k in tqdm(range(self.K)):
        s[0] = s_init
        s, u = self.landing_scp(s[0], 3, s_init, u_init)

        for k in range(self.N):
            s[k+1] =  self.dynamics(s[k], u[k])
        """
        if np.abs(s[0] - self.pad_trajectory[k, 0]) < tol and np.abs(s[1] - self.pad_trajectory[k, 1]) < tol and np.abs(s[4] - self.pad_trajectory[k, 4]) < tol:
            self.landed = True
            touchdowntime = time()
            touchdownvels[0] = s_mpc[k, 0, 2]
            touchdownvels[1] = s_mpc[k, 0, 3]
            touchdownvels[2] = s_mpc[k, 0, 5]
        """
        for k in range(self.N):
            total_control_cost += u[k].T @ self.R @ u[k]

        #u_init = np.concatenate([u_mpc[k, 1:], u_mpc[k, -1:]])
        #s_init = np.concatenate([s_mpc[k, 1:], self.dynamics(s_mpc[k, -1], u_mpc[k, -1]).reshape([1, -1])])

        # Plot trajectory and controls
        fig, ax = plt.subplots(1, 2, dpi=150, figsize=(15, 5))
        fig.suptitle('$N = {}$, '.format(self.N) + r'$N_\mathrm{SCP} = ' + '{}$'.format(self.N))

        #for t in range(self.T):
        #   ax[0].plot(s_mpc[t, :, 0], s_mpc[t, :, 1], '--*', color='k')
        ax[0].plot(s[:, 0], s[:, 1], '-')
        ax[0].set_xlabel(r'$x(t)$')
        ax[0].set_ylabel(r'$y(t)$')
        ax[0].axis('equal')

        ax[1].plot(u[:,  0], '-', label=r'$u_1(t)$')
        ax[1].plot(u[:,  1], '-', label=r'$u_2(t)$')
        ax[1].set_xlabel(r'$t$')
        ax[1].set_ylabel(r'$u(t)$')
        ax[1].legend()

        suffix = '_Nmpc={}_Nscp={}'.format(self.N, self.N_scp)
        plt.savefig('Figures/test_scp' + suffix + '.png', bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print('s shape', s.shape)
        print('u shape', u.shape)
        u_tmp[:-1] = u

        return s, u_tmp, total_control_cost, self.landed, touchdowntime, touchdownvels


class QC_controller_SCP_unconst(QC_controller_SCP):
    """
    Controller for a quadcopter without constraints
    """
    def __init__(self, quadcopter, state_dim, control_dim, P, Q, R, s_init, s_goal, T, dt):
        """
        Functionality
            Initialisation of a controller for a quadcopter using non-linear MPC

        Parameters
            quadcopter: quadcopter to be controlled
            s_init: initial state of the quadcopter
        """
        self.qc = quadcopter

        self.n = state_dim                  # state dimension
        self.m = control_dim                # control dimension
        self.P = P                          # terminal state cost matrix
        self.Q = Q                          # state cost matrix
        self.R = R                          # control cost matrix
        self.eps = 1e-3                     # SCP convergence tolerance
        self.s_init = s_init                # initial state
        self.s_goal = s_goal                # goal state
        self.T = T #s                       # simulation time
        self.dt = dt #s                     # sampling time
        self.K = int(self.T / self.dt) + 1  # number of steps
        self.N = int(self.T / self.dt)#self.K                    # MPC rollout steps
        self.N_scp = 300                      # Max. number of SCP interations

        self.dynamics = self.qc.discrete_dynamics_jnp

        self.landed = False

        self.timeline = None
        self.pad_trajectory = None

    def scp_iteration(self, s0, k0, s_prev, u_prev):
        A, B, c = ct.affinize(self.dynamics, s_prev[:-1], u_prev)
        A, B, c = np.array(A), np.array(B), np.array(c)

        s_cvx = cp.Variable((self.N + 1, self.n))
        u_cvx = cp.Variable((self.N , self.m))

        # Construction of the convex SCP sub-problem.
        costs = []
        constraints = []

        for k in range(self.N + 1):
            if k == 0:
                # initial condition
                constraints.append(s_cvx[k] == s0)

            if k == self.N:
                # terminal cost
                costs.append(cp.quad_form(s_cvx[k] - self.pad_trajectory[k], self.P))

            if k < self.N:
                # dynamics constraint
                constraints.append(A[k] @ s_cvx[k] + B[k] @ u_cvx[k] + c[k] == s_cvx[k+1])

                # stage cost
                costs.append(cp.quad_form(s_cvx[k] - self.pad_trajectory[k], self.Q))
                costs.append(cp.quad_form(u_cvx[k], self.R))

        objective = cp.sum(costs)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()

        if problem.status != 'optimal':
            raise RuntimeError('SCP solve failed. Problem status: ' + problem.status)
        s = s_cvx.value
        u = u_cvx.value
        J = problem.objective.value

        return s, u, J


class QC_controller_SCP_constr(QC_controller_SCP):
    """
    Controller for a quadcopter with constraints
        u_max: maximum torque of the rotors
        u_diff: maximum change of torque of the rotors
    """
    def __init__(self, quadcopter, state_dim, control_dim, P, Q, R, rs, ru, rT, s_init, s_goal, T, dt, u_max = np.inf, u_diff = np.inf):
        """
        Functionality
            Initialisation of a controller for a quadcopter using non-linear MPC

        Parameters
            quadcopter: quadcopter to be controlled
            s_init: initial state of the quadcopter
        """
        self.qc = quadcopter

        self.n = state_dim                  # state dimension
        self.m = control_dim                # control dimension
        self.P = P                          # terminal state cost matrix
        self.Q = Q                          # state cost matrix
        self.R = R                          # control cost matrix
        self.eps = 1e-3                     # SCP convergence tolerance
        self.s_init = s_init                # initial state
        self.s_goal = s_goal                # goal state
        self.T = T #s                       # simulation time
        self.dt = dt #s                     # sampling time
        self.K = int(self.T / self.dt) + 1  # number of steps
        self.N_mpc = 10                     # MPC rollout steps
        self.N_scp = 3                      # Max. number of SCP interations

        # Controller constraints
        self.rs = rs
        self.ru = ru
        self.rT = rT
        self.u_max = u_max
        self.u_diff = u_diff

        self.dynamics = self.qc.discrete_dynamics_jnp

        self.landed = False

        self.timeline = None
        self.pad_trajectory = None

    def scp_iteration(self, s0, k0, s_prev, u_prev):
        A, B, c = ct.affinize(self.dynamics, s_prev[:-1], u_prev)
        A, B, c = np.array(A), np.array(B), np.array(c)

        s_cvx = cp.Variable((self.N + 1, self.n))
        u_cvx = cp.Variable((self.N, self.m))

        # Construction of the convex SCP sub-problem.
        costs = []
        constraints = []

        for k in range(self.N + 1):
            # Global constraints

            if k == 0:
                # initial condition
                constraints.append(s_cvx[k] == s0)

            if k == self.N_mpc:
                # terminal cost
                costs.append(cp.quad_form(s_cvx[k] - self.pad_trajectory[ k], self.P))

            if k < self.N_mpc:
                # dynamics constraint
                constraints.append(A[k] @ s_cvx[k] + B[k] @ u_cvx[k] + c[k] == s_cvx[k+1])

                # stage cost
                costs.append(cp.quad_form(s_cvx[k] - self.pad_trajectory[k], self.Q))
                costs.append(cp.quad_form(u_cvx[k], self.R))

                # control contraints
                constraints.append(cp.abs(u_cvx[k]) <= self.u_max)
                constraints.append(cp.norm(u_cvx[k] - u_prev[k], 'inf') <= self.u_diff)

        objective = cp.sum(costs)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()

        if problem.status != 'optimal':
            raise RuntimeError('SCP solve failed. Problem status: ' + problem.status)
        s = s_cvx.value
        u = u_cvx.value
        J = problem.objective.value

        return s, u, J