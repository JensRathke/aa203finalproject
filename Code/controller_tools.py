#!/usr/bin/env python3

import numpy as np

def is_controllable(A, B, detailed_return=False):
    """
    Functionality
        Checks the controllability of a system defined by A and B

    Parameters
        A: matrix A
        B: matrix B
        detailed_return: if False, the return is only true or false / if True, all characteristics are returned

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

    gramian = CM.T.dot(CM)
    gramian_eigs = np.linalg.eigvals(gramian)

    if CM_rank == n:
        controllable = True

    if detailed_return:
        return controllable, CM_rank, CM_eigs, gramian_eigs
    else:
        return controllable

if __name__ == '__main__':
    A1 = np.array([[1, 0], [0, 2]])
    A2 = np.array([[1, 1], [0, 2]])
    B = np.array([[0, 1]]).T

    print(is_controllable(A1, B, False))
    print(is_controllable(A1, B, True))
    print(is_controllable(A2, B, False))
    print(is_controllable(A2, B, True))


