"""
Starter code for the problem "Cart-pole swing-up with limited actuation".

Autonomous Systems Lab (ASL), Stanford University
"""

from functools import partial

from animations import animate_cartpole

import cvxpy as cvx

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import numpy as np
from plotting import *
from animation import *


from tqdm import tqdm


def plot_states( t, s, filename, plot_titles=["", "", "", "", "", ""], y_labels=["", "", "", "", "", ""]):
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

@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0))
def affinize(f, s, u):
    """Affinize the function `f(s, u)` around `(s, u)`.

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
    A : jax.numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `s`.
    B : jax.numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `u`.
    c : jax.numpy.ndarray
        The offset term in the first-order Taylor expansion of `f` at `(s, u)`
        that sums all vector terms strictly dependent on the nominal point
        `(s, u)` alone.
    """
    # PART (b) ################################################################
    # INSTRUCTIONS: Use JAX to affinize `f` around `(s, u)` in two lines.
    A, B = jax.jacobian(f,(0,1))(s,u)
    c = f(s,u) - A @ s - B @ u
    # END PART (b) ############################################################
    return A, B, c


def solve_swingup_scp(f, s0, s_goal, N, P, Q, R, u_max, ρ, eps, max_iters):
    """Solve the cart-pole swing-up problem via SCP.

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
    for k in range(N):
        s[k+1] = fd(s[k], u[k])

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    J = np.zeros(max_iters + 1)
    J[0] = np.inf
    for i in (prog_bar := tqdm(range(max_iters))):
        s, u, J[i + 1] = scp_iteration(f, s0, s_goal, s, u, N,
                                       P, Q, R, u_max, ρ)
        dJ = np.abs(J[i + 1] - J[i])
        prog_bar.set_postfix({'objective change': '{:.5f}'.format(dJ)})
        if dJ < eps:
            converged = True
            print('SCP converged after {} iterations.'.format(i))
            break
    if not converged:
        print('SCP did not converge!')
        #raise RuntimeError('SCP did not converge!')
    J = J[1:i+1]
    return s, u, J


def scp_iteration(f, s0, s_goal, s_prev, u_prev, N, P, Q, R, u_max, ρ):
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
    A, B, c = affinize(f, s_prev[:-1], u_prev)
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


def cartpole(s, u):
    """Compute the cart-pole state derivative."""
    mp = 1.     # pendulum mass
    mc = 4.     # cart mass
    L = 1.      # pendulum length
    g = 9.81    # gravitational acceleration

    x, θ, dx, dθ, _, _ = s
    sinθ, cosθ = jnp.sin(θ), jnp.cos(θ)
    h = mc + mp*(sinθ**2)
    ds = jnp.array([
        dx,
        dθ,
        0, #(mp*sinθ*(L*(dθ**2) + g*cosθ) + u[0]) / h,
       (u[0]) / 1.0 - 9.8 ,# -((mc + mp)*g*sinθ + mp*L*(dθ**2)*sinθ*cosθ + u[0]*cosθ) / (h*L)
       0,
       0
    ])
    return ds
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
            -(u1)* jnp.sin(phi) /2.5,            #-(u1)* jnp.sin(phi) /2.5, #-(t1 + t2) * np.sin(phi) / self.m,
             (u1)*jnp.cos(phi) / 2.5 - 9.81, #-1.,# - self.g, #(t1 + t2) * np.cos(phi) / self.m - self.g,

            omega,
            u2 /Izz,
            #0.,
            #0.,

        ])
        return ds

def discretize(f, dt):
    """Discretize continuous-time dynamics `f` via Runge-Kutta integration."""

    def integrator(s, u, dt=dt):
        k1 = dt * f(s, u)
        k2 = dt * f(s + k1 / 2, u)
        k3 = dt * f(s + k2 / 2, u)
        k4 = dt * f(s + k3, u)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integrator


# Define constants
n = 6                                # state dimension
m = 2                              # control dimension
#s_goal = np.array([0, np.pi, 0, 0])  # desired upright pendulum state
s_goal = np.array([0, 0, 0, 0, 0, 0])  # desired upright pendulum state
s_init = np.array([4., 60., 0., 0., -np.pi / 4, -1.])
#s_init = np.array([4., 60., 0., 0., 0., 0.])
s0 = s_init
#s0 = np.array([6, 4, 0, 0, 0, 0])          # initial downright pendulum state
dt = 0.1                             # discrete time resolution
T = 30.                              # total simulation time
P = 1e2*np.eye(n)                    # terminal state cost matrix
#P[1][1] = P[1][1] * 10
#P = 1e2*np.eye(n)                    # terminal state cost matrix
#Q = np.diag([1., 1., 1e-3, 1e-3, 1e-3, 1e-3])  # state cost matrix
Q = jnp.diag(jnp.array([10., 10., 1., 1., 10., 1.]))
R = 1e-3*np.eye(m)                   # control cost matrix
ρ = 10000.                               # trust region parameter
u_max = 8000.                           # control effort bound
#eps = 5e-1                           # convergence tolerance
eps = 2.5e-1                           # convergence tolerance
max_iters = 100                      # maximum number of SCP iterations
enable_animate = True                      # flag for animation

# Initialize the discrete-time dynamics
#fd = jax.jit(discretize(cartpole, dt))
fd = jax.jit(discretize(dynamics_jnp,dt))

# Solve the swing-up problem with SCP
t = np.arange(0., T + dt, dt)
N = t.size -1
s, u, J = solve_swingup_scp(fd, s0, s_goal, N, P, Q, R, u_max, ρ,
                            eps, max_iters)

# Simulate open-loop control
for k in range(N):
    s[k+1] = fd(s[k], u[k])

# Plot state and control trajectories
fig, ax = plt.subplots(1, n + m, dpi=150, figsize=(15, 2))
plt.subplots_adjust(wspace=0.45)
labels_s = (r'$x(t)$', r'$y(t)$', r'$\dot{x}(t)$', r'$\dot{y}(t)$', r'$\phi(t)$', r'$\omega(t)$')
labels_u = (r'$u1(t)$',r'$u2(t)$')
plt.title('Q 2.5 d) Plots for Swing up constrained')
for i in range(n):
    ax[i].plot(t, s[:, i])
    ax[i].axhline(s_goal[i], linestyle='--', color='tab:orange')
    ax[i].set_xlabel(r'$t$')
    ax[i].set_ylabel(labels_s[i])
for i in range(m):
    ax[n + i].plot(t[:-1], u[:, i])
    ax[n + i].axhline(u_max, linestyle='--', color='tab:orange')
    ax[n + i].axhline(-u_max, linestyle='--', color='tab:orange')
    ax[n + i].set_xlabel(r'$t$')
    ax[n + i].set_ylabel(labels_u[i])
plt.savefig('cartpole_swingup_constrained.png',
            bbox_inches='tight')

# Plot cost history over SCP iterations
fig, ax = plt.subplots(1, 1, dpi=150, figsize=(8, 5))
ax.semilogy(J)
ax.set_xlabel(r'SCP iteration $i$')
ax.set_ylabel(r'SCP cost $J(\bar{x}^{(i)}, \bar{u}^{(i)})$')
plt.title('Q 2.5 d) Cost history')
plt.savefig('cartpole_swingup_constrained_cost.png',
            bbox_inches='tight')
plt.show()

sg = np.zeros((t.size, 6))
#sg[:, 1] = self.qcopter.h

# Animate the solution
if enable_animate:

   plot_states(t, s, "test_SCP_s", ["x", "y", "dx", "dy", "theta", "omega"])
   plot_controls(t[0:N], u, "test_SCP_u", ["T1", "T2"])
   animate(t, s, sg, "test_SCP")
  #fig, ani = animate_cartpole(t, s[:, 0], s[:, 1])
  #ani.save('cartpole_swingup_constrained.mp4', writer='ffmpeg')
  #plt.show()
