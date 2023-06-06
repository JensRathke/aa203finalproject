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
    # 2D Quadcopter Simulations
    ################################################################################
    T = 20 #s
    dt = 0.05 #s 

    n = 6 # state dimension
    m = 2 # control dimension

    s_init = np.array([4., 45., 0., 0., -0.1 * np.pi, 0.])
    ################################################################################
    # Single Simulation
    ################################################################################
    quadcopter = QuadcopterPlanar(2.5, 1.0, 0.5, 0.7, dt, 0.4)
    
    P = 1e2 * np.diag(np.array([5., 5., 1., 5., 10., 1.]))
    Q = np.diag(np.array([5., 5., 2., 30., 40., 10.]))
    R = 0.1 * np.eye(m)
    
    rs = np.inf
    ru = 40.0
    rdu = 5.0
    rT = np.inf
    N_scp = 3
    N_mpc = 20
    known_pad_dynamics = True
    simulate_wind = False
    filename = "000_test"

    controller = QC_controller_nlMPC_unconst(quadcopter, n, m, P, Q, R, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    # sim.simulate()

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    # sim.simulate()

    if False:
        exit()

    ################################################################################
    # Deterministic dynamics
    ################################################################################
    quadcopter = QuadcopterPlanar(2.5, 1.0, 0.5, 0.7, dt, 0.0)
    
    P = 1e2 * np.diag(np.array([5., 5., 1., 5., 10., 1.]))
    Q = np.diag(np.array([5., 5., 2., 30., 40., 10.]))
    R = 0.1 * np.eye(m)
    
    """
    Simulation of an unconstraint non-linear MPC
    """
    N_scp = 3
    N_mpc = 20
    known_pad_dynamics = True
    simulate_wind = False
    filename = "101_det_unconstr_3_20"

    controller = QC_controller_nlMPC_unconst(quadcopter, n, m, P, Q, R, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    rs = np.inf
    ru = 40.0
    rdu = 5.0
    rT = np.inf
    N_scp = 3
    N_mpc = 20
    known_pad_dynamics = True
    simulate_wind = False
    filename = "102_det_constr40_3_20"

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    rs = np.inf
    ru = 20.0
    rdu = 5.0
    rT = np.inf
    N_scp = 3
    N_mpc = 20
    known_pad_dynamics = True
    simulate_wind = False
    filename = "103_det_constr20_3_20"

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an unconstraint non-linear MPC
    """
    N_scp = 3
    N_mpc = 20
    known_pad_dynamics = False
    simulate_wind = False
    filename = "104_det_unconstr_3_20_unknownpad"

    controller = QC_controller_nlMPC_unconst(quadcopter, n, m, P, Q, R, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    rs = np.inf
    ru = 20.0
    rdu = 5.0
    rT = np.inf
    N_scp = 3
    N_mpc = 20
    known_pad_dynamics = False
    simulate_wind = False
    filename = "105_det_constr20_3_20_unknownpad"

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    rs = np.inf
    ru = 20.0
    rdu = 5.0
    rT = np.inf
    N_scp = 3
    N_mpc = 5
    known_pad_dynamics = True
    simulate_wind = False
    filename = "106_det_constr20_3_5"

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    rs = np.inf
    ru = 20.0
    rdu = 5.0
    rT = np.inf
    N_scp = 3
    N_mpc = 10
    known_pad_dynamics = True
    simulate_wind = False
    filename = "107_det_constr20_3_10"

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    rs = np.inf
    ru = 20.0
    rdu = 5.0
    rT = np.inf
    N_scp = 3
    N_mpc = 20
    known_pad_dynamics = True
    simulate_wind = False
    filename = "108_det_constr20_3_20"

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    rs = np.inf
    ru = 20.0
    rdu = 5.0
    rT = np.inf
    N_scp = 3
    N_mpc = 40
    known_pad_dynamics = True
    simulate_wind = False
    filename = "109_det_constr20_3_40"

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    rs = np.inf
    ru = 20.0
    rdu = 5.0
    rT = np.inf
    N_scp = 1
    N_mpc = 20
    known_pad_dynamics = True
    simulate_wind = False
    filename = "110_det_constr20_1_20"

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    rs = np.inf
    ru = 20.0
    rdu = 5.0
    rT = np.inf
    N_scp = 10
    N_mpc = 20
    known_pad_dynamics = True
    simulate_wind = False
    filename = "111_det_constr20_10_20"

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()


    ################################################################################
    # Noisy dynamics
    ################################################################################
    quadcopter = QuadcopterPlanar(2.5, 1.0, 0.5, 0.7, dt, 0.4)

    """
    Simulation of an unconstraint non-linear MPC
    """
    N_scp = 3
    N_mpc = 20
    known_pad_dynamics = True
    simulate_wind = False
    filename = "201_noise_unconstr_3_20"

    controller = QC_controller_nlMPC_unconst(quadcopter, n, m, P, Q, R, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    rs = np.inf
    ru = 20.0
    rdu = 5.0
    rT = np.inf
    N_scp = 3
    N_mpc = 10
    known_pad_dynamics = True
    simulate_wind = False
    filename = "202_noise_constr20_3_10"

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    rs = np.inf
    ru = 20.0
    rdu = 5.0
    rT = np.inf
    N_scp = 3
    N_mpc = 20
    known_pad_dynamics = True
    simulate_wind = False
    filename = "203_noise_constr20_3_20"

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    rs = np.inf
    ru = 20.0
    rdu = 5.0
    rT = np.inf
    N_scp = 3
    N_mpc = 20
    known_pad_dynamics = False
    simulate_wind = False
    filename = "204_noise_constr20_3_20_unknownpad"

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()


    ################################################################################
    # Deterministic dynamics with wind
    ################################################################################
    quadcopter = QuadcopterPlanar(2.5, 1.0, 0.5, 0.7, dt, 0.0)

    """
    Simulation of an unconstraint non-linear MPC
    """
    N_scp = 3
    N_mpc = 20
    known_pad_dynamics = True
    simulate_wind = True
    filename = "301_wind_unconstr_3_20"

    controller = QC_controller_nlMPC_unconst(quadcopter, n, m, P, Q, R, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    rs = np.inf
    ru = 40.0
    rdu = 5.0
    rT = np.inf
    N_scp = 3
    N_mpc = 10
    known_pad_dynamics = True
    simulate_wind = True
    filename = "302_wind_constr40_3_10"

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    rs = np.inf
    ru = 20.0
    rdu = 5.0
    rT = np.inf
    N_scp = 3
    N_mpc = 20
    known_pad_dynamics = True
    simulate_wind = True
    filename = "303_wind_constr20_3_20"

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    rs = np.inf
    ru = 40.0
    rdu = 5.0
    rT = np.inf
    N_scp = 3
    N_mpc = 20
    known_pad_dynamics = True
    simulate_wind = True
    filename = "304_wind_constr40_3_20"

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()

    """
    Simulation of an constraint non-linear MPC
    """
    rs = np.inf
    ru = 40.0
    rdu = 5.0
    rT = np.inf
    N_scp = 3
    N_mpc = 40
    known_pad_dynamics = True
    simulate_wind = True
    filename = "305_wind_constr40_3_40"

    controller = QC_controller_nlMPC_constr(quadcopter, n, m, P, Q, R, rs, ru, rT, rdu, s_init, N_mpc, N_scp, T, dt, filename, known_pad_dynamics, simulate_wind)
    sim = SimulationPlanar(quadcopter, controller, T, dt, k_buffer=N_mpc, output_filename=filename)
    sim.simulate()
    
    ################################################################################
    # 3D Quadcopter Simulations
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


    # s_goal = np.array([0., quadcopter.h, 0., 0., 0. , 0.])

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