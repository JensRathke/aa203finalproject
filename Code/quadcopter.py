#!/usr/bin/env python3

import numpy as np

class QuadcopterPlanar:
    """ Planar Quadcopter """
    def __init__(self, mass = 450, len_rotor_arm = 4.6, cabin_radius = 1.5):
        """
        Functionality
            Quadcopter class
        
        Parameters
            mass: mass of the quadcopter (default 450 kg)
            len_rotor_arm: length of a rotor arm (default 4.6 m)
            cabin_radius: radius of the cabin (default 1.5 m)
        """
        self.m = mass
        self.l = len_rotor_arm
        self.r = cabin_radius
        self.g = 9.81 # m/s²

    def quadcopter_dynamics(self, s, u):
        """
        Functionality
            Quadcopter dynamics
        
        Parameters
            s: state (x, dx, y, dy, phi, omega)
            u: control input (t1, t2)

        Returns
            derivative of the state with respect to time
        """
        x, dx, y, dy, phi, omega = s
        t1, t2 = u

        Ixx = (2 * self.m * (self.r^2) / 5) + 2 * self.m * (self.l^2) 
        Iyy = (2 * self.m * (self.r^2) / 5) + 2 * self.m * (self.l^2)
        Izz = (2 * self.m * (self.r^2) / 5) + 4 * self.m * (self.l^2)

        ds = np.array([
            dx,
            (-(t1 + t2) * np.sin(phi)) / self.m,
            dy,
            ((t1 + t2) * np.cos(phi)) / self.m - self.g,
            omega,
            ((t2 - t1) * self.l) / Izz
        ])

        return ds
    
class QuadcopterCubic:
    """ Cubic Quadcopter """
    def __init__(self, mass = 450, len_rotor_arm = 4.6, cabin_radius = 1.5):
        """
        Functionality
            Quadcopter class
        
        Parameters
            mass: mass of the quadcopter (default 450 kg)
            len_rotor_arm: length of a rotor arm (default 4.6 m)
            cabin_radius: radius of the cabin (default 1.5 m)
        """
        self.m = mass
        self.l = len_rotor_arm
        self.r = cabin_radius
        self.g = 9.81 # m/s²

    def quadcopter_dynamics(self, s, u):
        """
        Functionality
            Quadcopter dynamics
        
        Parameters
            s: state (x, dx, y, dy, phi, omega)
            u: control input (t1, t2)

        Returns
            derivative of the state with respect to time
        """
        x, dx, y, dy, phi, omega = s
        t1, t2 = u

        Ixx = (2 * self.m * (self.r^2) / 5) + 2 * self.m * (self.l^2) 
        Iyy = (2 * self.m * (self.r^2) / 5) + 2 * self.m * (self.l^2)
        Izz = (2 * self.m * (self.r^2) / 5) + 4 * self.m * (self.l^2)

        ds = np.array([
            dx,
            (-(t1 + t2) * np.sin(phi)) / self.m,
            dy,
            ((t1 + t2) * np.cos(phi)) / self.m - self.g,
            omega,
            ((t2 - t1) * self.l) / Izz
        ])

        return ds

if __name__ == "__main__":
    pass