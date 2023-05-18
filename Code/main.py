#!/usr/bin/env python3

import numpy as np

from simulation import *
from quadcopter import *
from spacecraft import *
from controller_test import *
from controller_iLQR import *
from controller_SCP import *
from controller_MPC import *
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

    s_init = np.array([4., 6., 2., 2., -np.pi / 4, -1.])

    select_controller = 3
    
    if select_controller == 1:
        print("iLQR controller")
        controller = PQcopter_controller_iLQR(quadcopter, s_init)
    elif select_controller == 2:
        print("SCP controller")
        controller = PQcopter_controller_SCP(quadcopter, s_init)
    elif select_controller == 3:
        print("MPC controller")
        controller = PQcopter_controller_MPC(quadcopter, s_init)
    else:
        print("Test controller")
        controller = PQcopter_controller_test(quadcopter, s_init)

    controller.land()



