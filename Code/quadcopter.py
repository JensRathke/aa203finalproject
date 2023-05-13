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
            s: state (x, y, dx, dy, psi, omega)
            u: control input (t1, t2)

        Returns
            derivative of the state with respect to time
        """
        x, y, dx, dy, psi, omega = s
        t1, t2 = u

        Ixx = (2 * self.m * (self.r^2) / 5) + 2 * self.m * (self.l^2) 
        Iyy = (2 * self.m * (self.r^2) / 5) + 2 * self.m * (self.l^2)
        Izz = (2 * self.m * (self.r^2) / 5) + 4 * self.m * (self.l^2)

        ds = np.array([
            dx,
            (-(t1 + t2) * np.sin(psi)) / self.m,
            dy,
            ((t1 + t2) * np.cos(psi)) / self.m - self.g,
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
            s: state (x, y, z, dx, dy, dz, phi, theta, psi, dphi, dtheta, dpsi)
                x = the inertial (north) position of the quadrotor along i_head
                y = the inertial (east) position of the quadrotor along j_head
                z = the altitude of the aircraft measured along -k_head
                dx = the body frame velocity measured along i_head
                dy = the body frame velocity measured along j_head
                dz = the body frame velocity measured along k_head
                phi = the roll angle
                theta = the pitch angle
                psi = the yaw angle
                dphi = the roll rate measured along i_head
                dtheta = the pitch rate measured along j_head
                dpsi = the yaw rate measured along k_head
            u: control input (t1, t2, t3, t4)

        Returns
            derivative of the state with respect to time
        """
        x, y, z, dx, dy, dz, phi, theta, psi, dphi, dtheta, dpsi = s
        t1, t2, t3, t4 = u

        Ixx = (2 * self.m * (self.r^2) / 5) + 2 * self.m * (self.l^2) 
        Iyy = (2 * self.m * (self.r^2) / 5) + 2 * self.m * (self.l^2)
        Izz = (2 * self.m * (self.r^2) / 5) + 4 * self.m * (self.l^2)

        ds = np.array([
            dx,
            (-(t1 + t2) * np.sin(phi)) / self.m,
            dy,
            ((t1 + t2) * np.cos(phi)) / self.m - self.g,
            dtheta,
            ((t2 - t1) * self.l) / Izz
        ])

        return ds

if __name__ == "__main__":
    pass