"""
Starter code for the problem "Cart-pole swing-up".

Autonomous Systems Lab (ASL), Stanford University
"""

import time

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt


import numpy as np
from numpy import linalg as LA
from plotting import *
from animation import *

from scipy.integrate import odeint

def plot_states(t, s, filename, plot_titles=["", "", "", "", "", ""], y_labels=["", "", "", "", "", ""]):
      """
      Functionality
          Plot quadcopter states

      Parameters
          t: time
          s: state trajectory (x, y, dx, dy, psi, omega)
          filename: name of the output file without file-extension
      """
      plot_3x2(filename, t, s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5], "trajectory", plot_titles=plot_titles, ylabels=y_labels)

def plot_controls(t, u, filename, plot_titles=["", ""], y_labels=["", ""]):
    """
    Functionality
        Plot a quadcopter trajectory

    Parameters
        t: time
        u: controls (t1, t2)
        filename: name of the output file without file-extension
    """
    plot_1x2(filename, t, u[:, 0], u[:, 1], "controls", plot_titles=plot_titles, ylabels=y_labels)

def animate( t, s, sg, filename):
        """
        Functionality
            Animate a quadcopter trajectory

        Parameters
            t: time
            s: state trajectory (x, y, dx, dy, psi, omega)
            sg: goal state trajectory (x, y, dx, dy, psi, omega)
            filename: name of the output file without file-extension
        """
        animate_planar_quad(filename, t, s[:, 0], s[:, 1], s[:, 4], sg[:, 0], sg[:, 1], sg[:, 4], 3., 2., 2.)#self.l, self.r, self.h)

def linearize(f, s, u):
    """Linearize the function `f(s, u)` around `(s, u)`.

    Arguments
    ---------
    f : callable
        A nonlinear function with call signature `f(s, u)`.
    s : numpy.ndarray
        The state (1-D).
    u : numpy.ndarray
        The control input (1-D).

    Returns
    -------
    A : numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `s`.
    B : numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `u`.
    """
    # WRITE YOUR CODE BELOW ###################################################
    # INSTRUCTIONS: Use JAX to compute `A` and `B` in one line.
    A, B = jax.jacobian(f, (0,1))(s, u)
    ###########################################################################
    return A, B


def ilqr(f, s0, s_goal, N, Q, R, QN, eps=1e-3, max_iters=1000):
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
        A, B = jax.vmap(linearize, in_axes=(None, 0, 0))(f, s_bar[:-1], u_bar)
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


        # Hyeong's code:
        # P_T = QN
        # P_k = P_T
        # p_T= q_T = QN.T@ (s_bar[-1]-s_goal)
        # p_k=p_T
        # q_k=q_T
        # r_T = R.T@ u_bar[-1]
        # r_k= r_T
        # #backward recursion
        # for j in range(N-1, -1, -1):
        #     q_k = Q.T @ (s_bar[j]-s_goal)
        #     r_k = R.T @ u_bar[j]
        #     hxt = q_k + A[j].T @ p_k
        #     hut = r_k + B[j].T @ p_k
        #     Hxxt = Q + A[j].T @ P_k @ A[j]
        #     Huut = R + B[j].T @ P_k @ B[j]
        #     Hxut = A[j].T @ P_k @ B[j]

        #     #K_k= -1.0*LA.pinv(Huut)@Hxut.transpose()
        #     Y[j]= -1.0 * LA.pinv(Huut) @ Hxut.T
        #     #k_k = -1.0 * LA.pinv(Huut)@hut
        #     y[j] = -1.0 * LA.pinv(Huut) @ hut
        #     #p_k = hxt + Hxxt@k_k
        #     p_k = hxt + Hxut @ y[j]
        #     #P_k = Hxxt +Hxut@K_k
        #     P_k = Hxxt + Hxut @ Y[j]
        #     """
        #     if np.isnan(P_k).any():
        #         import ipdb; ipdb.set_trace()
        #     """

        # #forward pass
        # #roll out
        # for k in range(N):
        #     du[k] = y[k] + Y[k] @ ds[k]
        #     ds[k+1] = f(s_bar[k] + ds[k], u_bar[k] + du[k]) - s_bar[k+1]
        #     #s_bar[k+1] = s_bar[k+1] + ds[k+1]
        #     #u_bar[k] = u_bar[k] + du[k]
        # s_bar+=ds
        # u_bar+=du
        #######################################################################
        #print(np.max(np.abs(du)))
        if np.max(np.abs(du)) != 0.0 and np.max(np.abs(du)) < eps:
           converged = True
           print(i, np.max(np.abs(du)))
           break
    if not converged:
        print('iLQR did not converge!')
        #raise RuntimeError('iLQR did not converge!')
    return s_bar, u_bar, Y, y


def dynamics_jnp(s, u):
        """
        Functionality
            Continuous-time quadcopter dynamics

        Parameters
            s: state (x, y, dx, dy, phi, omega)
            u: control input (u1, u2)

        Returns
            derivative of the state with respect to time
        """
        x, y, dx, dy, phi, omega = s
        #x, y, dx, dy = s

        #x, y, dx, dy, phi, omega
        # t1, t2 = u
        u1, u2 = u
        #m, l, r ,h
        #2.5, 1.0, .5, 0.7
        Izz = (2. * 2.5 * (0.5 ** 2.) / 5.) + 4. * 2.5 * (1.0** 2.)
        """
        ds = jnp.array([
            dx,
            dy,
            -(u1)* jnp.sin(phi) / self.m, #-(t1 + t2) * np.sin(phi) / self.m,
            (u1)*jnp.cos(phi) / self.m, #-1.,# - self.g, #(t1 + t2) * np.cos(phi) / self.m - self.g,
            omega,
            (u2) / self.Izz #(t1 - t2) * self.l / (2 * self.Izz)
        ])
        """
        """
        ds = jnp.array([
            dx,
            dy,
             -(u1)* phi / self.m,#jnp.sin(phi) / self.m, #-(t1 + t2) * np.sin(phi) / self.m,
            (u1)*(1.-jnp.abs(phi)) / self.m , #- self.g, #jnp.cos(phi) / self.m - self.g, #(t1 + t2) * np.cos(phi) / self.m - self.g,
            omega,
            (u2) / self.Izz #(t1 - t2) * self.l / (2 * self.Izz)
        ])
        """

        """
        0, # -(u1)* phi / self.m,#jnp.sin(phi) / self.m, #-(t1 + t2) * np.sin(phi) / self.m,
        (u1) / 1.0 - 9.8, #jnp.cos(phi) / self.m - self.g, #(t1 + t2) * np.cos(phi) / self.m - self.g,
        0,
        0
        """
        ds = jnp.array([
            dx,
            dy,
            #0,
            #(u1) / 2.5 - 9.81,
            #-u1 * phi/2.5,
            #(u1)*(1. - jnp.abs(phi)) / 2.5 - 9.81,
            -(u1+u2)* jnp.sin(phi) /2.5,            #-(u1)* jnp.sin(phi) /2.5, #-(t1 + t2) * np.sin(phi) / self.m,
             (u1+u2)*jnp.cos(phi) / 2.5 - 9.81, #-1.,# - self.g, #(t1 + t2) * np.cos(phi) / self.m - self.g,

            omega,
            -(u1-u2) /Izz,
            #0.,
            #0.,

        ])
        return ds

# Define constants
n = 6                                       # state dimension
m = 2                                       # control dimension
Q = np.diag(np.array([10., 10., 100., 100.,100., 100.]))   # state cost matrix
#Q = jnp.diag(jnp.array([1e-3,1e-3, 1e-3, 1e-3, 1., 1.]))
R = 1e1*np.eye(m)                          # control cost matrix
QN = 1e3*np.eye(n)                          # terminal state cost matrix
s0 = np.array([4., 50., 0., 0.,-np.pi / 4, 0.])             # initial state
s_goal = np.array([0., 0., 0., 0., 0.,0.])      # goal state
T = 20.                                     # simulation time
dt = 0.1                                    # sampling time
#animate = True                             # flag for animation
closed_loop = False                         # flag for closed-loop control

# Initialize continuous-time and discretized dynamics
#f = jax.jit(cartpole)
f = jax.jit(dynamics_jnp)
fd = jax.jit(lambda s, u, dt=dt: s + dt*f(s, u))

# Compute the iLQR solution with the discretized dynamics
print('Computing iLQR solution ... ', end='', flush=True)
start = time.time()
t = np.arange(0., T, dt)
N = t.size - 1
s_bar, u_bar, Y, y = ilqr(fd, s0, s_goal, N, Q, R, QN)
print('done! ({:.2f} s)'.format(time.time() - start), flush=True)

# Simulate on the true continuous-time system
print('Simulating ... ', end='', flush=True)
start = time.time()
s = np.zeros((N + 1, n))
u = np.zeros((N, m))
s[0] = s0
for k in range(N):
    # PART (d) ################################################################
    # INSTRUCTIONS: Compute either the closed-loop or open-loop value of
    # `u[k]`, depending on the Boolean flag `closed_loop`.
    if closed_loop:
        u[k] = u_bar[k] + y[k] + Y[k] @ (s[k]-s_bar[k])
        #raise NotImplementedError()
    else:  # do open-loop control
        u[k] = u_bar[k]
        #raise NotImplementedError()
    ###########################################################################
    s[k+1] = odeint(lambda s, t: f(s, u[k]), s[k], t[k:k+2])[1]
print('done! ({:.2f} s)'.format(time.time() - start), flush=True)

# Plot cost history over SCP iterations
fig, ax = plt.subplots(1, 1, dpi=150, figsize=(8, 5))
"""
ax.semilogy(J)
ax.set_xlabel(r'SCP iteration $i$')
ax.set_ylabel(r'SCP cost $J(\bar{x}^{(i)}, \bar{u}^{(i)})$')
plt.title('SCP Cost history')
plt.savefig('scp_cost.png',
            bbox_inches='tight')
plt.show()
"""
sg = np.zeros((t.size, 6))
#sg[:, 1] = self.qcopter.h
enable_animate = True
# Animate the solution
if enable_animate:

   plot_states(t, s, "test_iLQR_s", ["x", "y", "dx", "dy", "theta", "omega"])
   plot_controls(t[0:N], u, "test_iLQR_u", ["U1", "U2"])
   animate(t, s, sg, "test_iLQR")
  #fig, ani = animate_cartpole(t, s[:, 0], s[:, 1])
  #ani.save('cartpole_swingup_constrained.mp4', writer='ffmpeg')
  #plt.show()
