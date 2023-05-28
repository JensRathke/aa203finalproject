#!/usr/bin/env python3

import numpy as np
import numpy as np
import jax
import jax.numpy as jnp

from functools import partial
from typing import Callable, NamedTuple

def linearize(f, s, u):
    """
    Functionality
        Linearize the function `f(s, u)` around `(s, u)`.

    Parameters
        f : callable
            A nonlinear function with call signature `f(s, u)`.
        s : numpy.ndarray
            The state (1-D).
        u : numpy.ndarray
            The control input (1-D).

    Returns
        A : numpy.ndarray
            The Jacobian of `f` at `(s, u)`, with respect to `s`.
        B : numpy.ndarray
            The Jacobian of `f` at `(s, u)`, with respect to `u`.
    """
    # Option A
    # A, B = jax.jacfwd(f, argnums=(0, 1))(s, u)
    
    # Option B
    A, B = jax.jacobian(f, (0, 1))(s, u)

    return A, B

@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0))
def affinize(f, s, u):
    """
    Affinize the function `f(s, u)` around `(s, u)`.
    """
    A, B  = jax.jacobian(f, (0, 1))(s, u)
    c = f(s, u) - A @ s - B @ u

    return A, B, c

def discretize(f, dt):
    """
    Discretize continuous-time dynamics `f` via Runge-Kutta integration.
    """

    def integrator(s, u, dt=dt):
        k1 = dt * f(s, u)
        k2 = dt * f(s + k1 / 2, u)
        k3 = dt * f(s + k2 / 2, u)
        k4 = dt * f(s + k3, u)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integrator

def is_controllable(A, B, detailed_return=False):
    """
    Functionality
        Checks the controllability of a system defined by A and B

    Parameters
        A: matrix A
        B: matrix B
        detailed_return: if False, the return is only True or False / if True, all characteristics are returned

    Returns
        controllability [True / False]
        controllability [True / False], CM_rank, CM_eigs, A_eigs, gramian_eigs
    """
    controllable = False
    n = A.shape[0]

    # Build Characteristic Matrix
    CM = B
    for k in range(1, n):
        CM = np.hstack((CM, (A ** k) @ B))

    # Check rank and eigenvalues
    CM_rank = np.linalg.matrix_rank(CM)         # -> full rank => all state variables are controllable
    CM_eigs = np.linalg.eigvals(CM)
    A_eigs = np.linalg.eigvals(A)               # -> stability of the open-loop system

    # Check gramian and its eigenvectors and eigenvalues
    gramian = CM.T.dot(CM)
    gramian_eigs = np.linalg.eig(gramian)       # -> eigenvectors corresponding to the largest eigenvalues of the gramian identify the most controllable direction of the system

    if CM_rank == n:
        controllable = True

    if detailed_return:
        return controllable, CM_rank, CM_eigs, A_eigs, gramian_eigs
    else:
        return controllable

if __name__ == '__main__':
    A1 = np.array([[1, 0], [0, 2]])
    A2 = np.array([[1, 1], [0, 2]])
    B = np.array([[0, 1]]).T

    print(is_controllable(A1, B))
    print(is_controllable(A1, B, True))
    print(is_controllable(A2, B))
    print(is_controllable(A2, B, True))

