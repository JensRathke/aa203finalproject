#!/usr/bin/env python3

import numpy as np
import cvxpy as cvx

from scipy.linalg import solve_discrete_are

from quadcopter import *

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
        self.qcopter = qcopter
        
        self.n = 6                                  # state dimension
        self.m = 2                                  # control dimension
        self.Q = np.diag(np.array([1., 1., 1., 1., 1., 1.]))   # state cost matrix
        self.R = 10 * np.eye(self.m)                     # control cost matrix
        self.s_init = s_init                        # initial state
        self.s_goal = np.array([0., self.qcopter.h, 0., 0., 0. , 0.])      # goal state
        self.T = 30  # s                           # simulation time
        self.dt = 0.1 # s                           # sampling time
        self.N = 3                                  # rollout steps
        self.rs = 5.0
        self.ru = 0.5
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
                constraints.append(x_cvx[k] == x0)

            if k == N:
                # terminal cost
                costs.append(cvx.quad_form(x_cvx[k], P))

                # terminal contraints
                constraints.append(cvx.norm(x_cvx[k], 'inf') <= rf)

            if k <= N and k > 0:
                # dynamics constraint
                constraints.append(A @ x_cvx[k-1] + B @ u_cvx[k-1] == x_cvx[k])

            if k < N:
                # stage cost
                costs.append(cvx.quad_form(x_cvx[k], Q))
                costs.append(cvx.quad_form(u_cvx[k], R))

                # state contraints
                constraints.append(cvx.norm(x_cvx[k], 'inf') <= rx)

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
        Ps = (np.eye(self.n), np.zeros((self.n, self.n)))
        titles = (r'$P = I$', r'$P = P_\mathrm{DARE}$')
        x0s = (self.s_init, np.array([0., 10., 0., 0., 0. , 0.]))

        fig, ax = plt.subplots(2, len(Ps), dpi=150, figsize=(10, 8),
                            sharex='row', sharey='row')
        
        print("Ps", Ps)
        print("titles", titles)
        print("x0s", x0s)
        print("T", self.T)

        for i, (P, title) in enumerate(zip(Ps, titles)):
            for x0 in x0s:
                x = np.copy(x0)
                x_mpc = np.zeros((self.T, self.N + 1, self.n))
                u_mpc = np.zeros((self.T, self.N, self.m))
                for t in range(1, self.T):
                    if P[0, 0] == 0:
                        A, B, P = self.linearize_penalize(self.qcopter.dynamics_jnp, x_mpc[t-1, 0], u_mpc[t-1, 0])
                    else:
                        A, B, _ = self.linearize_penalize(self.qcopter.dynamics_jnp, x_mpc[t-1, 0], u_mpc[t-1, 0])

                    x_mpc[t], u_mpc[t], status = self.mpc_rollout(x, A, B, P, self.Q, self.R, self.N, self.rs, self.ru, self.rT)
                    if status == 'infeasible':
                        x_mpc = x_mpc[:t]
                        u_mpc = u_mpc[:t]
                        break
                    print(x_mpc)
                    x = A@x + B@u_mpc[t, 0, :]
                    ax[0, i].plot(x_mpc[t, :, 0], x_mpc[t, :, 1], '--*', color='k')
                ax[0, i].plot(x_mpc[:, 0, 0], x_mpc[:, 0, 1], '-o')
                ax[1, i].plot(u_mpc[:, 0], '-o')
            ax[0, i].set_title(title)
            ax[0, i].set_xlabel(r'$x_{k,1}$')
            ax[1, i].set_xlabel(r'$k$')
        ax[0, 0].set_ylabel(r'$x_{k,2}$')
        ax[1, 0].set_ylabel(r'$u_k$')
        fig.savefig('Figures/P3.2_mpc_feasibility_sim.png', bbox_inches='tight')
        plt.show()

class PQcopter_controller_nlMPC():
    """ Controller for a planar quadcopter using non-linear MPC """
    def __init__(self, qcopter: QuadcopterPlanar, s_init):
        """
        Functionality
            Initialisation of a controller for a planar quadcopter using non-linear MPC

        Parameters
            qcopter: quadcopter to be controlled
            s_init: initial state of the quadcopter
        """
        self.qcopter = qcopter

        self.n = 6                                  # state dimension
        self.m = 2                                  # control dimension
        self.Q = np.diag(np.array([1., 1., 1., 1., 1., 1.]))   # state cost matrix
        self.R = 10 * np.eye(self.m)                     # control cost matrix
        self.s_init = s_init                        # initial state
        self.s_goal = np.array([0., self.qcopter.h, 0., 0., 0. , 0.])      # goal state
        self.T = 30  # s                           # simulation time
        self.dt = 0.1 # s                           # sampling time
        self.N = 3                                  # rollout steps
        self.rs = 5.0
        self.ru = 0.5
        self.rT = np.inf