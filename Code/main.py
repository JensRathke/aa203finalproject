#!/usr/bin/env python3

import numpy as np

from time import time

from simulation import *
from quadcopter import *
from spacecraft import *
from controller_iLQR import *
from controller_SCP import *
from controller_MPC import *
from controller_nlMPC import *
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
    ################################################################################
    # 2D Quadcopter Simultions
    ################################################################################
    T = 20 #s
    dt = 0.05 #s

    n = 6
    m = 2

    quadcopter = QuadcopterPlanar(2.5, 1.0, 0.5, 0.7, dt, 0.0)

    s_init = np.array([4., 45., 0., 0., -0.1 * np.pi, 0.])
    s_goal = np.array([0., quadcopter.h, 0., 0., 0. , 0.])

    # P = 1e2 * np.diag(np.array([1., 1., 0., 10., 10., 0.]))
    # Q = np.diag(np.array([5., 5., 2., 40., 50., 10.]))
    # R = 0.1 * np.eye(m)

    

    # select_controller = 0

    # if select_controller == 1:
    #     print("iLQR controller")
    #     controller = QC_controller_iLQR(quadcopter, s_init)
    # elif select_controller == 2:
    #     print("SCP controller")
    # elif select_controller == 3:
    #     print("MPC controller with linearization")
    #     controller = PQcopter_controller_MPC(quadcopter, s_init)
    # elif select_controller == 4:
    #     print("non-linear MPC controller")
    #     controller = QC_controller_nlMPC_unconst(quadcopter, n, m, P, Q, R, s_init, s_goal, T, dt)
    # elif select_controller == 5:
    #     print("SCP controller")
    #     controller = QC_controller_SCP_unconst(quadcopter, n, m, P, Q, R, s_init, s_goal, T, dt)


    # if select_controller != 0:
    #     controller.land()

    """
    Simulation of an unconstraint non-linear MPC
    """
    P = 1e2 * np.diag(np.array([1., 1., 0., 10., 10., 0.]))
    Q = np.diag(np.array([5., 5., 2., 40., 50., 10.]))
    R = 0.1 * np.eye(m)
    N_mpc = 10
    N_scp = 3
    known_pad_dynamics = True

    controller1 = QC_controller_nlMPC_unconst(quadcopter, n, m, P, Q, R, s_init, N_mpc, N_scp, T, dt, known_pad_dynamics)
    sim1 = SimulationPlanar(quadcopter, controller1, T, dt, output_filename="test_nlMPC_uncontraint")
    # sim1.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    P = 1e2 * np.diag(np.array([5., 5., 1., 5., 10., 1.]))
    Q = np.diag(np.array([5., 5., 2., 30., 40., 10.]))
    R = 0.1 * np.eye(m)
    rs = np.inf
    ru = 20.0
    rdu = 5.0
    rT = np.inf
    N_mpc = 20
    N_scp = 3
    known_pad_dynamics = True

    controller2 = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, known_pad_dynamics)
    sim2 = SimulationPlanar(quadcopter, controller2, T, dt, output_filename="test_nlMPC_constraint")
    sim2.simulate()

    """
    Simulation of an iLQR controller
    """
    QN = 1e3*np.eye(n)
    Q = np.diag(np.array([10., 10., 100., 100.,100., 100.]))
    R = 1e1*np.eye(m)

    controller3 = QC_controller_iLQR(quadcopter, n, m, QN, Q, R, s_init, s_goal, T, dt)
    sim3 = SimulationPlanar(quadcopter, controller3, T, dt, output_filename="test_iLQR")
    # sim3.simulate()


    ################################################################################
    # 3D Quadcopter Simultions
    ################################################################################

    # Test with a 3D quadcopter
    # quadcopter2 = QuadcopterCubic(2.5, 1.0, .5, 0.7)

    # s_init = np.array([4., 0., 60., 0., 0., 0., 0., -np.pi / 4, 0., 0., 0., 0.])
    # s_goal = np.array([0., 0., quadcopter2.h, 0., 0., 0. , 0., 0., 0. , 0., 0., 0.])

    # P = 1e2 * np.eye(12)
    # Q = np.diag(jnp.array([10., 10., 10., 1., 1., 1., 10., 10., 10., 1., 1., 1.]))
    # R = 0.1 * jnp.eye(4)

    # controller = QC_controller_nlMPC(quadcopter2, 12, 4, P, Q, R, rs, ru, rT, s_init, s_goal)

    # controller.land()



