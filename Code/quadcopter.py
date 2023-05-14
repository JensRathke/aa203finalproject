#!/usr/bin/env python3

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from plotting import *
from animation import *

class QuadcopterPlanar:
    """ Planar Quadcopter """
    def __init__(self, mass = 450, len_rotor_arm = 3.6, cabin_radius = 1.2, h_centre_of_mass = 1.5):
        """
        Functionality
            Initialisation of a planar quadcopter
        
        Parameters
            mass: mass of the quadcopter (default 450 kg)
            len_rotor_arm: length of a rotor arm (default 3.6 m)
            cabin_radius: radius of the cabin (default 1.25 m)
            h_centre_of_mass: height of the center of mass above skid surface (default 1.5 m)
        """
        self.m = mass # kg
        self.l = len_rotor_arm # m
        self.r = cabin_radius # m
        self.h = h_centre_of_mass # m
        self.g = 9.81 # m/s²

    def dynamics(self, s, u):
        """
        Functionality
            Quadcopter dynamics
        
        Parameters
            s: state (x, y, dx, dy, phi, omega)
            u: control input (t1, t2)

        Returns
            derivative of the state with respect to time
        """
        x, y, dx, dy, phi, omega = s
        t1, t2 = u

        Ixx = (2. * self.m * (self.r ** 2.) / 5.) + 2. * self.m * (self.l ** 2.) 
        Iyy = (2. * self.m * (self.r ** 2.) / 5.) + 2. * self.m * (self.l ** 2.)
        Izz = 1 # (2. * self.m * (self.r ** 2.) / 5.) + 4. * self.m * (self.l ** 2.)

        ds = jnp.array([
            dx,
            (-(t1 + t2) * jnp.sin(phi)) / self.m,
            dy,
            ((t1 + t2) * jnp.cos(phi)) / self.m - self.g,
            omega,
            ((t2 - t1) * self.l) / Izz
        ])

        return ds
    
    def animate(self, t, s, sg, filename):
        """
        Functionality
            Animate a quadcopter trajectory
        
        Parameters
            t: time
            s: state trajectory (x, y, dx, dy, psi, omega)
            sg: goal state trajectory (x, y, dx, dy, psi, omega)
            filename: name of the output file without file-extension
        """
        animate_planar_quad(filename, t, s[:, 0], s[:, 1], s[:, 4], sg[:, 0], sg[:, 1], sg[:, 4], self.l, self.r, self.h)

    def plot_trajectory(self, t, s, filename):
        """
        Functionality
            Plot a quadcopter trajectory
        
        Parameters
            t: time
            s: state trajectory (x, y, dx, dy, psi, omega)
            filename: name of the output file without file-extension
        """
        plot_3x2(filename, t, s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5], "iLQR trajectory")

    def plot_controls(self, t, u, filename):
        """
        Functionality
            Plot a quadcopter trajectory
        
        Parameters
            t: time
            u: controls (t1, t2)
            filename: name of the output file without file-extension
        """
        plot_1x2(filename, t, u[:, 0], u[:, 1], "iLQR controls")
            
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

    def dynamics(self, s, u):
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
    testcopter = QuadcopterPlanar()

    t = np.linspace(0, 10, 101)

    s = np.zeros((101, 6))
    s[:, 1] = np.linspace(4 + testcopter.h, 0 + testcopter.h, 101)
    sg = np.zeros((101, 6))
    sg[:, 4] = 0.1 * np.sin(t)
    sg[:, 0] = 0.3 * np.sin(t)

    print(s[:, 1])
    print(s[:, 4])
    print(sg[:, 4])

    testcopter.animate(t, s, sg, "Animations/test")
