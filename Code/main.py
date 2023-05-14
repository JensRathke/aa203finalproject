#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from simulation import *
from quadcopter import *
from spacecraft import *
from controller import *
from plotting import *
from animation import *

if __name__ == '__main__':

    quadcopter = QuadcopterPlanar()

    s_init = np.array((1., 5., 0., 0., 0.25 * np.pi, 0.))

    controller = PQcopter_controller_iLQR(quadcopter, s_init)

    controller.land()



