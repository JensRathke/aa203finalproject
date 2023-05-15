#!/usr/bin/env python3

import numpy as np

from simulation import *
from quadcopter import *
from spacecraft import *
from controller import *
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

    quadcopter = QuadcopterPlanar()

    s_init = jnp.array((1., 10., 0., 0., 0.1 * np.pi, 0.))
    use_ilqr=True
    if use_ilqr:
        controller = PQcopter_controller_iLQR(quadcopter, s_init)
    else:
        controller = PQcopter_controller_SCP(quadcopter, s_init)

    controller.land()



