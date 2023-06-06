#!/usr/bin/env python3

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def dynamics(t, x):
    J1 = 1700
    J2 = 2000
    J3 = 1400
    omega = np.array([x[3], x[4], x[5]])
    K = controller()
    u = -K * x
    phi_dot = omega[0]
    theta_dot = omega[1]
    psi_dot = omega[2]

    omega1_dot = 1 / J1 * ((J2 - J3) * omega[1] * omega[2] + u[0])
    omega2_dot = 1 / J2 * ((J3 - J1) * omega[0] * omega[2] + u[1])
    omega3_dot = 1 / J3 * ((J1 - J2) * omega[1] * omega[0] + u[2])

    return np.array([phi_dot, theta_dot, psi_dot, omega1_dot, omega2_dot, omega3_dot])

def controller():
    mu = 398600.4418e9      # gravitational constant
    r = 6378.14e3 + 500e3   # radius of earth plus flight height


if __name__ == "__main__":
    pass