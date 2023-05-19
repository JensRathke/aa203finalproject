#!/usr/bin/env python3

import numpy as np
import numpy as np
import jax
import jax.numpy as jnp

from typing import Callable, NamedTuple

class RK4Integrator(NamedTuple):
    """Discrete time dynamics from time-invariant continuous time dynamics using a 4th order Runge-Kutta method."""
    ode: Callable
    dt: float

    @jax.jit
    def __call__(self, x, u, k):
        k1 = self.dt * self.ode(x, u)
        k2 = self.dt * self.ode(x + k1 / 2, u)
        k3 = self.dt * self.ode(x + k2 / 2, u)
        k4 = self.dt * self.ode(x + k3, u)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def linearize(f, s, u):
        A, B = jax.jacobian(f, (0, 1))(s, u)
        return A, B

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
        controllability [True / False], CM_rank, CM_eigs, gramian_eigs
    """
    controllable = False
    CM = B

    n = A.shape[0]

    for k in range(1, n):
        CM = np.hstack((CM, (A ** k) @ B))

    CM_rank = np.linalg.matrix_rank(CM)
    CM_eigs = np.linalg.eigvals(CM)
    A_eigs = np.linalg.eigvals(A)

    gramian = CM.T.dot(CM)
    gramian_eigs = np.linalg.eigvals(gramian)

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

