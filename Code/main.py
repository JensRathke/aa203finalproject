#!/usr/bin/env python3

import numpy as np

from time import time

from simulation import *
from quadcopter import *
from spacecraft import *
from controller_test import *
from controller_iLQR import *
from controller_SCP import *
from controller_MPC import *
from controller_nlMPC import *
from controller_a_scp import *
from plotting import *
from animation import *

"""
******
Simulation of different controllers to land a quadcopter on a moving landing pad.
******

States
======
state of the quadcopter in 2D: (x, y, dx, dy, phi, omega)
state of the quadcopter in 3D: (x, y, z, dx, dy, dz, phi, theta, psi, dphi, dtheta, dpsi)

state of the landing pad in 2D: (x, y, 0, 0, phi, 0)
state of the landing pad in 3D: (x, y, z, 0, 0, 0, phi, theta, psi, 0, 0, 0)
"""

if __name__ == '__main__':

    quadcopter = QuadcopterPlanar(2.5, 1.0, .5, 0.7)

    s_init = np.array([4., 45., 0., 0., -0.1 * np.pi, -1.])
    s_goal = np.array([0., quadcopter.h, 0., 0., 0. , 0.])
    #s_init = np.array([4., 45., 0., 0., 0., 0.])

    T = 20 #s
    dt = 0.05 #s

    n = 6
    m = 2

    P = 1e2 * np.diag(np.array([1., 1., 0., 10., 10., 0.]))
    Q = np.diag(np.array([5., 5., 2., 40., 50., 10.]))
    R = 0.1 * np.eye(m)

    u_max = 40.0
    u_diff = 10.0

    rs = 5.0
    ru = 0.1
    rT = np.inf

    select_controller = 0
    select_controller = 0

    if select_controller == 1:
        print("iLQR controller")
        controller = PQcopter_controller_iLQR(quadcopter, s_init)
    elif select_controller == 2:
        print("SCP controller")
        controller = PQcopter_controller_SCP(quadcopter, s_init)
    elif select_controller == 3:
        print("MPC controller with linearization")
        controller = PQcopter_controller_MPC(quadcopter, s_init)
    elif select_controller == 4:
        print("non-linear MPC controller")
        controller = QC_controller_nlMPC_unconst(quadcopter, n, m, P, Q, R, s_init, s_goal, T, dt)
    elif select_controller == 5:
        print("SCP controller")
        controller = QC_controller_SCP_unconst(quadcopter, n, m, P, Q, R, s_init, s_goal, T, dt)
    elif select_controller == 9:
        print("Test controller")
        controller = PQcopter_controller_test(quadcopter, s_init)

    if select_controller != 0:
        controller.land()

    controller1 = QC_controller_nlMPC_unconst(quadcopter, n, m, P, Q, R, s_init, s_goal, T, dt)

    sim1 = SimulationPlanar(quadcopter, controller1, T, dt, output_filename="test_nlMPC_uncontraint")
    sim1.simulate()

    controller2 = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, s_init, s_goal, T, dt, u_max, u_diff)

    sim2 = SimulationPlanar(quadcopter, controller2, T, dt, output_filename="test_nlMPC_constraint")
    sim2.simulate()
    sim2.simulate()

    controller3 = QC_controller_SCP_unconst(quadcopter, n, m, P, Q, R, s_init, s_goal, T, dt)
    sim3 = SimulationPlanar(quadcopter, controller3, T, dt, output_filename="test_SCP_uncontraint")
    sim3.simulate()


    # Test with a 3D quadcopter
    # quadcopter2 = QuadcopterCubic(2.5, 1.0, .5, 0.7)

    # s_init = np.array([4., 0., 60., 0., 0., 0., 0., -np.pi / 4, 0., 0., 0., 0.])
    # s_goal = np.array([0., 0., quadcopter2.h, 0., 0., 0. , 0., 0., 0. , 0., 0., 0.])

    # P = 1e2 * np.eye(12)
    # Q = np.diag(jnp.array([10., 10., 10., 1., 1., 1., 10., 10., 10., 1., 1., 1.]))
    # R = 0.1 * jnp.eye(4)

    # controller = QC_controller_nlMPC(quadcopter2, 12, 4, P, Q, R, rs, ru, rT, s_init, s_goal)

    # controller.land()



