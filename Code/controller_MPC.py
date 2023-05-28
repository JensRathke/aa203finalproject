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
                constraints.append(A @ x_cvx[k-1] + B @ u_cvx[k-1] == x_cvx[k])

            if k < N:
                # stage cost
                costs.append(cp.quad_form(x_cvx[k] - self.s_goal, Q))
                costs.append(cp.quad_form(u_cvx[k], R))

                # state contraints
                # constraints.append(cvx.norm(x_cvx[k] - x_cvx[k-1], 'inf') <= rx)

                # control contraints
                constraints.append(cp.norm(u_cvx[k], 'inf') <= ru)

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
        x_mpc = jnp.zeros((self.T, self.N + 1, self.n))
        x_mpc = x_mpc.at[0, 0].set(self.s_init)
        u_mpc = jnp.zeros((self.T, self.N, self.m))

        P = jnp.eye(self.n)

        # Initialize continuous-time and discretized dynamics
        f = jax.jit(self.qc.dynamics_jnp)
        fd = jax.jit(ct.discretize(self.qc.dynamics_jnp, self.dt))

        for t in range(1, self.T):
            # A, B, _ = self.linearize_penalize(self.qc.dynamics_jnp, x_mpc[t-1, 0], u_mpc[t-1, 0])
            # A, B = jax.vmap(ct.linearize, in_axes=(None, 0, 0))(f, x_mpc[t-1, :-1], u_mpc[t-1])

            # A, B = ct.linearize(self.qc.discrete_dynamics_jnp, x_mpc[t-1, 0], u_mpc[t-1, 0])

            print(">>>", x_mpc[t-1, :-1], u_mpc[t-1])
            A, B, c = self.affinize(fd, x_mpc[t-1, :-1], u_mpc[t-1])
            A, B = np.array(A[0]), np.array(B[0])

            print("x1:", x)
            x_mpc[t], u_mpc[t], status = self.mpc_rollout(x, A, B, P, self.Q, self.R, self.N, self.rs, self.ru, self.rT)
            print(x_mpc[t], u_mpc[t], status)
            if status == 'infeasible':
                x_mpc = x_mpc[:t]
                u_mpc = u_mpc[:t]
                break
            print("A, x, B, u:", A, x, B, u_mpc[t, 0])
            x = A @ x + B @ u_mpc[t, 0]
            print("x3:", x)

        x_values = jnp.zeros((x_mpc.shape[0], self.N + 1))
        y_values = jnp.zeros((x_mpc.shape[0], self.N + 1))

        for index in range(x_mpc.shape[0]):
            x_values[index] = x_mpc[index, :, 0]
            y_values[index] = x_mpc[index, :, 1]

        self.qc.animate(t_line, x_mpc[:, 0], x_mpc[:, 0], "test_MPC")
        plot_trajectory("test_MPC_traj", x_values, y_values)


class QC_controller_nlMPC():
    """ Controller for a planar quadcopter using non-linear MPC """
    def __init__(self, quadcopter: QuadcopterPlanar, state_dim, control_dim, P, Q, R, rs, ru, rT, s_init, s_goal):
        """
        Functionality
            Initialisation of a controller for a planar quadcopter using non-linear MPC

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
        self.T = 20  # s                    # simulation time
        self.dt = 0.1 # s                   # sampling time
        self.K = int(self.T / self.dt) + 1  # number of steps
        self.N_mpc = 10                     # MPC rollout steps
        self.N_scp = 3                      # Max. number of SCP interations
        self.rs = rs
        self.ru = ru
        self.rT = rT

        self.dynamics = self.qc.discrete_dynamics_jnp

        self.airborne = True

    def landing_scp(self, s0, s_init = None, u_init = None, convergence_error = False):
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
            s, u, J[iteration + 1] = self.scp_iteration(s0, s, u)

            dJ = np.abs(J[iteration + 1] - J[iteration])
            if dJ < self.eps:
                converged = True
                break

        if not converged and convergence_error:
            raise RuntimeError('SCP did not converge!')

        return s, u
    
    def scp_iteration(self, s0, s_prev, u_prev):
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
                costs.append(cp.quad_form(s_cvx[k] - self.s_goal, self.P))

            if k <= self.N_mpc and k > 0:
                # dynamics constraint
                constraints.append(A[k-1] @ s_cvx[k-1] + B[k-1] @ u_cvx[k-1] + c[k-1] == s_cvx[k])

            if k < self.N_mpc:
                # stage cost
                costs.append(cp.quad_form(s_cvx[k] - self.s_goal, self.Q))
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

    def land(self):
        s_mpc = np.zeros((self.K, self.N_mpc + 1, self.n))
        u_mpc = np.zeros((self.K, self.N_mpc, self.m))

        s = np.copy(self.s_init)

        total_time = time()
        total_control_cost = 0.0

        s_init = None
        u_init = None

        for t in tqdm(range(self.K)):
            s_mpc[t], u_mpc[t] = self.landing_scp(s, s_init, u_init)

            s = self.dynamics(s_mpc[t, 0], u_mpc[t, 0])

            total_control_cost += u_mpc[t, 0].T @ self.R @ u_mpc[t, 0]

            u_init = np.concatenate([u_mpc[t, 1:], u_mpc[t, -1:]])
            s_init = np.concatenate([s_mpc[t, 1:], self.dynamics(s_mpc[t, -1], u_mpc[t, -1]).reshape([1, -1])])

        total_time = time() - total_time
        print('Total elapsed time:', total_time, 'seconds')
        print('Total control cost:', total_control_cost)

        # Plot trajectory and controls
        # fig, ax = plt.subplots(1, 2, dpi=150, figsize=(15, 5))
        # fig.suptitle('$N = {}$, '.format(self.N_mpc) + r'$N_\mathrm{SCP} = ' + '{}$'.format(self.N_scp))

        # for t in range(self.T):
        #     ax[0].plot(s_mpc[t, :, 0], s_mpc[t, :, 1], '--*', color='k')
        # ax[0].plot(s_mpc[:, 0, 0], s_mpc[:, 0, 1], '-')
        # ax[0].set_xlabel(r'$x(t)$')
        # ax[0].set_ylabel(r'$y(t)$')
        # ax[0].axis('equal')

        # ax[1].plot(u_mpc[:, 0, 0], '-', label=r'$u_1(t)$')
        # ax[1].plot(u_mpc[:, 0, 1], '-', label=r'$u_2(t)$')
        # ax[1].set_xlabel(r'$t$')
        # ax[1].set_ylabel(r'$u(t)$')
        # ax[1].legend()

        # suffix = '_Nmpc={}_Nscp={}'.format(self.N_mpc, self.N_scp)
        # plt.savefig('Figures/test_nlmpc' + suffix + '.png', bbox_inches='tight')
        # plt.show()
        # plt.close(fig)

        t_line = np.arange(0., self.K * self.dt, self.dt)

        # Plot trajectory
        self.qc.plot_trajectory(t_line, s_mpc[:, 0], "test_nlMPC_trajectory")

        # Plot states
        self.qc.plot_states(t_line, s_mpc[:, 0], "test_nlMPC_states")
        
        # Plot states
        self.qc.plot_controls(t_line, u_mpc[:, 0], "test_nlMPC_controls")

        # Create animation
        self.qc.animate(t_line, s_mpc[:, 0], s_mpc[:, 0], "test_nlMPC")
