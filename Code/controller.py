#!/usr/bin/env python3

class PQcopter_controller_iLQR():
    """ Controller for a planar quadcopter using iLQR """
    def __init__(self, qcopter, s_init):
        """
        Functionality
            Initialisation of a controller for a planar quadcopter using iLQR
        
        Parameters
            qcopter: quadcopter to be controlled
            s_init: initial state of the quadcopter
        """
        self.qcopter = qcopter
        self.s_init = s_init


if __name__ == "__main__":
    pass