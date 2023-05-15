#!/usr/bin/env python3

import numpy as np
import jax
import jax.numpy as jnp
import time

from scipy.integrate import odeint

from quadcopter import *

# class PQcopter_controller_LQR():
#     """ Controller for a planar quadcopter using LQR """
#     def __init__(self, qcopter, s_init):
#         """
#         Functionality
#             Initialisation of a controller for a planar quadcopter using iLQR
        
#         Parameters
#             qcopter: quadcopter to be controlled
#             s_init: initial state of the quadcopter
#         """
#         self.qcopter = qcopter
#         self.s_init = s_init

    




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
        self.Q = np.diag(np.array([10., 10., 2., 2., 10., 2.]))   # state cost matrix
        self.R = 1e-2*np.eye(self.m)                     # control cost matrix
        self.QN = 1e2*np.eye(self.n)                     # terminal state cost matrix
        self.s_init = s_init                        # initial state
        self.s_goal = np.array([0., self.qcopter.h, 0., 0., 0. , 0.])      # goal state
        self.T = 30.                                # simulation time
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
            A, B = jax.vmap(self.linearize, in_axes=(None, 0, 0))(f, s_bar[:-1], u_bar)
            A, B = np.array(A), np.array(B)

            # PART (c) ############################################################
            # INSTRUCTIONS: Update `Y`, `y`, `ds`, `du`, `s_bar`, and `u_bar`.
            k = N - 1
            s_tilde = np.zeros((N + 1, n))
            u_tilde = np.zeros((N, m))
            q = np.zeros((N + 1, 1, n))
            r = np.zeros((N + 1, m))
            P = np.zeros((N + 1, Q.shape[0], Q.shape[1]))
            p = np.zeros((N + 1, n))
            hs = np.zeros((N, n))
            hu = np.zeros((N, m))
            Hss = np.zeros((N, Q.shape[0], Q.shape[1]))
            Huu = np.zeros((N, R.shape[0], R.shape[1]))
            Hsu = np.zeros((N, n, m))

            q[N] = s_bar[N].T @ QN - s_goal.T @ QN
            P[N] = Q
            p[N] = q[N]

            print("iteration: ", iteration)

            while k >= 0:
                q[k] = s_bar[k].T @ Q - s_goal.T @ Q
                r[k] = u_bar[k].T @ R
                
                # Approxmate terms for eta_k, h_s_k, h_u_k, H_ss_k, H_uu_k and H_su_k
                hs[k] = q[k] + A[k].T @ p[k+1]
                hu[k] = r[k] + B[k].T @ p[k+1]
                Hss[k] = Q + A[k].T @ P[k+1] @ A[k]
                Huu[k] = R + B[k].T @ P[k+1] @ B[k]
                Hsu[k] = A[k].T @ P[k+1] @ B[k]

                # Recursively computation of p_k, P_k, y_k, Y_k and beta_k
                Y[k] = -1 * np.linalg.inv(Huu[k]) @ Hsu[k].T
                y[k] = -1 * np.linalg.inv(Huu[k]) @ hu[k]
                P[k] = Hss[k] + Hsu[k] @ Y[k]
                p[k] = hs[k] + Hsu[k] @ y[k]

                k -= 1

            for k in range(0, N-1):
                # Rollout for stilde_k+1 and utilde_k
                u_tilde[k] = y[k] + Y[k] @ s_tilde[k]
                ds[k] = f(s_bar[k] + s_tilde[k], u_bar[k] + u_tilde[k])
                du[k] = (u_bar[k+1] - u_bar[k]) * self.dt
                # s_tilde[k+1] = s_bar[k] + s_tilde[k] + dt * ds[k] - s_bar[k+1]
                s_tilde[k+1] = f(s_bar[k] + s_tilde[k], u_bar[k] + u_tilde[k]) - s_bar[k+1]
                
                # Update (s_bar, u_bar)
                s_bar[k] = s_bar[k] + s_tilde[k]
                u_bar[k] = u_bar[k] + u_tilde[k]

            print("max du", np.max(np.abs(du)))
            #######################################################################

            if np.max(np.abs(du)) < eps:
                converged = True
                break
        if not converged:
            raise RuntimeError('iLQR did not converge!')
        return s_bar, u_bar, Y, y
    
    def land(self):
        # Initialize continuous-time and discretized dynamics
        f = jax.jit(self.qcopter.dynamics)
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
        sg[:, 1] = self.qcopter.h

        self.qcopter.plot_trajectory(t, s, "Figures/test_iLQR_s")
        self.qcopter.plot_controls(t[0:N], u, "Figures/test_iLQR_u")
        self.qcopter.animate(t, s, sg, "Animations/test_iLQR")

if __name__ == "__main__":
    pass