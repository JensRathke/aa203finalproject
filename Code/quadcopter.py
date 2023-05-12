#!/usr/bin/env python3

import numpy as np

def quadcopter_dynamics(s, u):
    x, vx, y, vy, phi, omega = s
    T1, T2 = u
    m = 450 # kg
    g = 9.81 # m/s²
    l = 4.6 # m (length of a rotor arm)
    r = 1.5 # m (radius of the cabin)
    Ixx = (2 * m * (r^2) / 5) + 2 * m * (l^2) 
    Iyy = (2 * m * (r^2) / 5) + 2 * m * (l^2)
    Izz = (2 * m * (r^2) / 5) + 4 * m * (l^2)

    ds = np.array([
        vx,
        (-(T1 + T2) * np.sin(phi)) / m,
        vy,
        ((T1 + T2) * np.cos(phi)) / m - g,
        omega,
        ((T2 - T1) * l) / Izz
    ])

    return ds

if __name__ == "__main__":
    pass