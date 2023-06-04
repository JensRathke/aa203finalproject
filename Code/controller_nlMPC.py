#!/usr/bin/env python3

import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import controller_tools as ct

from functools import partial
from time import time
from tqdm.auto import tqdm
from quadcopter import *
from plotting import *

filepath = "Figures/"

class QC_controller_nlMPC():
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
            s = np.zeros((self.N_mpc + 1, self.n))
            u = np.zeros((self.N_mpc, self.m))
            s[0] = s0
            for k in range(self.N_mpc):
                s[k+1] = self.dynamics(s[k], u[k])
        else:
            s = np.copy(s_init)
            u = np.copy(u_init)

        converged = False
        J = np.zeros(self.N_scp + 1)

        for iteration in range(self.N_scp):
            s, u, J[iteration + 1], A, B, c = self.mpc_rollout(s0, k0, s, u)

            dJ = np.abs(J[iteration + 1] - J[iteration])
            if dJ < self.eps:
                converged = True
                break

        if not converged and convergence_error:
            raise RuntimeError('SCP did not converge!')

        return s, u, A, B, c
    
    def mpc_rollout(self, s0, k0, s_prev, u_prev):
        raise NotImplementedError("Method must be overriden by a subclass of QC_controller_nlMPC")

    def land(self):
        s_mpc = np.zeros((self.K, self.N_mpc + 1, self.n))
        u_mpc = np.zeros((self.K, self.N_mpc, self.m))

        s = np.copy(self.s_init)

        tol = 0.1
        total_control_cost = 0.0
        touchdownvels = np.zeros(3)
        touchdowntime = 0

        s_init = None
        u_init = None
        
        for k in tqdm(range(self.K)):
            if self.landed == False:
                s_mpc[k], u_mpc[k], A, B, c = self.landing_scp(s, k, s_init, u_init)
            else:
                s_mpc[k] = self.pad_trajectory[k]
                u_mpc[k] = 0

            if self.wind == False:
                s = self.dynamics(s_mpc[k, 0], u_mpc[k, 0])
            else:
                s = A[0] @ s_mpc[k, 0] + B[0] @ u_mpc[k, 0] + c[0] + np.array([0.8, -0.05, 0.4, 0.05, 0.0, 0.0]) * np.clip(np.random.normal(0.1, 0.1), 0.0, 0.2) * np.clip(s_mpc[k, 0, 1], 0, 2)

            if self.landed == False and np.abs(s[0] - self.pad_trajectory[k, 0]) < tol and np.abs(s[1] - self.pad_trajectory[k, 1]) < tol and np.abs(s[4] - self.pad_trajectory[k, 4]) < tol:
                self.landed = True
                touchdowntime = self.timeline[k]
                if k > 0:
                    touchdownvels[0] = s_mpc[k, 0, 2] - (self.pad_trajectory[k, 2] - self.pad_trajectory[k-1, 2]) / self.dt
                    touchdownvels[1] = s_mpc[k, 0, 3] - (self.pad_trajectory[k, 3] - self.pad_trajectory[k-1, 3]) / self.dt
                    touchdownvels[2] = s_mpc[k, 0, 5] - (self.pad_trajectory[k, 5] - self.pad_trajectory[k-1, 5]) / self.dt

            total_control_cost += u_mpc[k, 0].T @ self.R @ u_mpc[k, 0]

            u_init = np.concatenate([u_mpc[k, 1:], u_mpc[k, -1:]])
            s_init = np.concatenate([s_mpc[k, 1:], self.dynamics(s_mpc[k, -1], u_mpc[k, -1]).reshape([1, -1])])

        # Plot trajectory and controls
        fig, ax = plt.subplots(1, 1, dpi=150)
        fig.suptitle('$N = {}$, '.format(self.N_mpc) + r'$N_\mathrm{SCP} = ' + '{}$'.format(self.N_scp))

        for k in range(self.K):
            ax.plot(s_mpc[k, :, 0], s_mpc[k, :, 1], '--', color='k', lw=0.5)
        ax.plot(s_mpc[:, 0, 0], s_mpc[:, 0, 1], '-', lw=1.0)
        ax.set_xlabel(r'$x(t)$')
        ax.set_ylabel(r'$y(t)$')
        ax.axis('equal')

        suffix = "_MPCrollout" #'_Nmpc={}_Nscp={}'.format(self.N_mpc, self.N_scp)
        plt.savefig(filepath + self.filename + suffix + '.png', bbox_inches='tight')
        plt.show()
        plt.close(fig)

        return s_mpc[:, 0], u_mpc[:, 0], total_control_cost, self.landed, touchdowntime, touchdownvels
    

class QC_controller_nlMPC_unconst(QC_controller_nlMPC):
    """
    Controller for a quadcopter without constraints
    """
    def __init__(self, quadcopter, state_dim, control_dim, P, Q, R, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics=False, wind=False):
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
        self.T = T #s                       # simulation time
        self.dt = dt #s                     # sampling time
        self.K = int(self.T / self.dt) + 1  # number of time steps
        self.N_scp = N_scp                  # Max. number of SCP interations
        self.N_mpc = N_mpc                  # MPC rollout steps
        self.filename = filename
        self.known_pad_dynamics = known_pad_dynamics
        self.wind = wind

        self.dynamics = self.qc.discrete_dynamics

        self.landed = False

        self.timeline = None
        self.pad_trajectory = None
        
        self.description = "unconstraint non-linear MPC"
        self.params = f"N_scp: {self.N_scp} / N_mpc: {self.N_mpc} / known_pad_dynamics: {self.known_pad_dynamics} / wind: {self.wind}"

    def mpc_rollout(self, s0, k0, s_prev, u_prev):
        A, B, c = ct.affinize(self.dynamics, s_prev[:-1], u_prev)
        A, B, c = np.array(A), np.array(B), np.array(c)

        s_cvx = cp.Variable((self.N_mpc + 1, self.n))
        u_cvx = cp.Variable((self.N_mpc, self.m))

        # Construction of the convex SCP sub-problem.
        costs = []
        constraints = []

        for k in range(self.N_mpc + 1):
            if k == 0:
                # initial condition
                constraints.append(s_cvx[k] == s0)

            if k == self.N_mpc:
                # terminal cost
                if self.known_pad_dynamics == True:
                    costs.append(cp.quad_form(s_cvx[k] - self.pad_trajectory[k0 + k], self.P))
                else:
                    costs.append(cp.quad_form(s_cvx[k] - self.pad_trajectory[k0], self.P))

            if k < self.N_mpc:
                # dynamics constraint
                constraints.append(A[k] @ s_cvx[k] + B[k] @ u_cvx[k] + c[k] == s_cvx[k+1])

                # stage cost
                if self.known_pad_dynamics == True:
                    costs.append(cp.quad_form(s_cvx[k] - self.pad_trajectory[k0 + k], self.Q))
                else:
                    costs.append(cp.quad_form(s_cvx[k] - self.pad_trajectory[k0], self.Q))
                costs.append(cp.quad_form(u_cvx[k], self.R))

        objective = cp.sum(costs)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()

        if problem.status != 'optimal':
            # raise RuntimeError('SCP solve failed. Problem status: ' + problem.status)
            pass
        s = s_cvx.value
        u = u_cvx.value
        J = problem.objective.value

        return s, u, J, A, B, c
    

class QC_controller_nlMPC_constr(QC_controller_nlMPC):
    """
    Controller for a quadcopter with constraints
        u_max: maximum torque of the rotors
        u_diff: maximum change of torque of the rotors
    """
    def __init__(self, quadcopter, state_dim, control_dim, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics=False, wind=False):
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
        self.T = T #s                       # simulation time
        self.dt = dt #s                     # sampling time
        self.K = int(self.T / self.dt) + 1  # number of time steps
        self.N_scp = N_scp                  # Max. number of SCP interations
        self.N_mpc = N_mpc                  # MPC rollout steps
        self.filename = filename
        self.known_pad_dynamics = known_pad_dynamics
        self.wind = wind

        # Controller constraints
        self.rs = rs
        self.ru = ru
        self.rT = rT
        self.rdu = rdu

        self.dynamics = self.qc.discrete_dynamics

        self.landed = False

        self.timeline = None
        self.pad_trajectory = None

        self.description = "constraint non-linear MPC"
        self.params = f"N_scp: {self.N_scp} / N_mpc: {self.N_mpc} / known_pad_dynamics: {self.known_pad_dynamics} / wind: {self.wind} / rs: {self.rs} / ru: {self.ru} / rT: {self.rT} / rdu: {self.rdu}"

    def mpc_rollout(self, s0, k0, s_prev, u_prev):
        A, B, c = ct.affinize(self.dynamics, s_prev[:-1], u_prev)
        A, B, c = np.array(A), np.array(B), np.array(c)

        s_cvx = cp.Variable((self.N_mpc + 1, self.n))
        u_cvx = cp.Variable((self.N_mpc, self.m))

        # Construction of the convex SCP sub-problem.
        costs = []
        constraints = []

        for k in range(self.N_mpc + 1):
            # Global constraints
            
            if k == 0:
                # initial condition
                constraints.append(s_cvx[k] == s0)

            if k == self.N_mpc:
                # terminal cost
                if self.known_pad_dynamics == True:
                    costs.append(cp.quad_form(s_cvx[k] - self.pad_trajectory[k0 + k], self.P))
                else:
                    costs.append(cp.quad_form(s_cvx[k] - self.pad_trajectory[k0], self.P))

            if k < self.N_mpc:
                # dynamics constraint
                constraints.append(A[k] @ s_cvx[k] + B[k] @ u_cvx[k] + c[k] == s_cvx[k+1])

                # stage cost
                if self.known_pad_dynamics == True:
                    costs.append(cp.quad_form(s_cvx[k] - self.pad_trajectory[k0 + k], self.Q))
                else:
                    costs.append(cp.quad_form(s_cvx[k] - self.pad_trajectory[k0], self.Q))
                costs.append(cp.quad_form(u_cvx[k], self.R))

                # control contraints
                constraints.append(cp.abs(u_cvx[k]) <= self.ru)
                constraints.append(u_cvx[k] >= 0)
                constraints.append(cp.norm(u_cvx[k] - u_prev[k], 'inf') <= self.rdu)    

        objective = cp.sum(costs)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()

        if problem.status != 'optimal':
            # raise RuntimeError('SCP solve failed. Problem status: ' + problem.status)
            pass
        s = s_cvx.value
        u = u_cvx.value
        J = problem.objective.value

        return s, u, J, A, B, c