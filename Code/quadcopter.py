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
        self.g = 9.81 # m/s²
        self.m = mass # kg
        self.l = len_rotor_arm # m
        self.r = cabin_radius # m
        self.h = h_centre_of_mass # m

        self.Ixx = (2. * self.m * (self.r ** 2.) / 5.) + 2. * self.m * (self.l ** 2.)
        self.Iyy = (2. * self.m * (self.r ** 2.) / 5.) + 2. * self.m * (self.l ** 2.)
        self.Izz = (2. * self.m * (self.r ** 2.) / 5.) + 4. * self.m * (self.l ** 2.)

    def dynamics(self, s, u):
        """
        Functionality
            Continuous-time quadcopter dynamics

        Parameters
            s: state (x, y, dx, dy, phi, omega)
            u: control input (u1, u2)

        Returns
            derivative of the state with respect to time
        """
        x, y, dx, dy, phi, omega = s
        t1, t2 = u

        ds = jnp.array([
            dx,
            dy,
            -(t1 + t2) * jnp.sin(phi) / self.m,
            (t1 + t2) * jnp.cos(phi) / self.m - self.g,
            omega,
            (t2 - t1) * self.l / self.Izz
        ])

        return ds
    
    def dynamics_jnp(self, s, u):
        """
        Functionality
            Continuous-time quadcopter dynamics

        Parameters
            s: state (x, y, dx, dy, phi, omega)
            u: control input (u1, u2)

        Returns
            derivative of the state with respect to time
        """
        x, y, dx, dy, phi, omega = s
        t1, t2 = u

        ds = jnp.array([
            dx,
            dy,
            -(t1 + t2) * jnp.sin(phi) / self.m,
            (t1 + t2) * jnp.cos(phi) / self.m - self.g,
            omega,
            (t2 - t1) * self.l / self.Izz
        ])

        return ds
    
    def dynamics_ode(self, s, t, u):
        """
        Functionality
            Continuous-time quadcopter dynamics

        Parameters
            s: state (x, y, dx, dy, phi, omega)
            t
            u: control input (u1, u2)

        Returns
            derivative of the state with respect to time
        """
        x, y, dx, dy, phi, omega = s
        t1, t2 = u

        ds = jnp.array([
            dx,
            dy,
            -(t1 + t2) * jnp.sin(phi) / self.m,
            (t1 + t2) * jnp.cos(phi) / self.m - self.g,
            omega,
            (t2 - t1) * self.l / self.Izz
        ])

        return ds
    
    def discrete_dynamics(self, s, u, dt = 0.1):
        """
        Functionality
            Discrete-time quadcopter dynamics

        Parameters
            s: state (x, y, dx, dy, phi, omega)
            u: control input (u1, u2)

        Returns
            derivative of the state with respect to time
        """
        x, y, dx, dy, phi, omega = s
        t1, t2 = u

        ds = jnp.array([
            dx,
            dy,
            -(t1 + t2) * jnp.sin(phi) / self.m,
            (t1 + t2) * jnp.cos(phi) / self.m - self.g,
            omega,
            (t2 - t1) * self.l / self.Izz
        ])

        return s + ds * dt
    
    def discrete_dynamics_jnp(self, s, u, dt = 0.1):
        """
        Functionality
            Discrete-time quadcopter dynamics

        Parameters
            s: state (x, y, dx, dy, phi, omega)
            u: control input (t1, t2)

        Returns
            derivative of the state with respect to time
        """
        x, y, dx, dy, phi, omega = s
        t1, t2 = u

        ds = jnp.array([
            dx,
            dy,
            -(t1 + t2) * jnp.sin(phi) / self.m,
            (t1 + t2) * jnp.cos(phi) / self.m - self.g,
            omega,
            (t2 - t1) * self.l / self.Izz
        ])

        return s + ds * dt

class QuadcopterCubic:
    """ Cubic Quadcopter """
    def __init__(self, mass = 450, len_rotor_arm = 3.6, cabin_radius = 1.2, h_centre_of_mass = 1.5):
        """
        Functionality
            Quadcopter class

        Parameters
            mass: mass of the quadcopter (default 450 kg)
            len_rotor_arm: length of a rotor arm (default 4.6 m)
            cabin_radius: radius of the cabin (default 1.5 m)
        """
        self.g = 9.81                   # gravity [m/s²]
        self.m = mass                   # mass [kg]
        self.l = len_rotor_arm          # length of rotor arm [m]
        self.r = cabin_radius           # cabin radius [m]
        self.h = h_centre_of_mass       # height of centre of mass [m]

        self.kf = 3.13e-5               # thrust coefficient [N s²]
        self.kM = 7.5e-7                # moment coefficient [Nm s²]
        self.kt = 0.1                   # aerodynamic thrust drag coefficient [N s/m]
        self.kr = 0.1                   # aerodynamic moment drag coefficient [Nm s]

        self.Ir = 6e-5                  # inertia of motor [kg m²]
        self.Ixx = (2. * self.m * (self.r ** 2.) / 5.) + 2. * self.m * (self.l ** 2.)   # moment of inertia [kg m²]
        self.Iyy = (2. * self.m * (self.r ** 2.) / 5.) + 2. * self.m * (self.l ** 2.)   # moment of inertia [kg m²]
        self.Izz = (2. * self.m * (self.r ** 2.) / 5.) + 4. * self.m * (self.l ** 2.)   # moment of inertia [kg m²]

    def dynamics_jnp(self, s, u):
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
        u1, u2, u3, u4 = u

        ds = jnp.array([
            dx,
            dy,
            dz,
            -1 / self.m * (self.kt * dx + u1 * (jnp.sin(phi) * jnp.sin(psi) + jnp.cos(phi) * jnp.cos(psi) * jnp.sin(theta))),
            -1 / self.m * (self.kt * dy + u1 * (jnp.sin(phi) * jnp.cos(psi) + jnp.cos(phi) * jnp.sin(psi) * jnp.sin(theta))),
            -1 / self.m * (self.kt * dz - self.m * self.g + u1 * (jnp.cos(phi) * jnp.cos(theta))),
            dphi,
            dtheta,
            dpsi,
            -1 / self.Ixx * (self.kr * dphi - self.l * u2 - self.Iyy * dtheta * dpsi + self.Izz * dtheta * dpsi),
            -1 / self.Iyy * (-self.kr * dtheta + self.l * u3 - self.Ixx * dphi * dpsi + self.Izz * dphi * dpsi),
            -1 / self.Izz * (u4 - self.kr * self.r + self.Ixx * dphi * dtheta - self.Iyy * dphi * dtheta)
        ])

        return ds
    
    def discrete_dynamics_jnp(self, s, u, dt = 0.1):
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
            dt: timestep

        Returns
            derivative of the state with respect to time
        """
        x, y, z, dx, dy, dz, phi, theta, psi, dphi, dtheta, dpsi = s
        u1, u2, u3, u4 = u

        ds = jnp.array([
            dx,
            dy,
            dz,
            -1 / self.m * (self.kt * dx + u1 * (jnp.sin(phi) * jnp.sin(psi) + jnp.cos(phi) * jnp.cos(psi) * jnp.sin(theta))),
            -1 / self.m * (self.kt * dy + u1 * (jnp.sin(phi) * jnp.cos(psi) + jnp.cos(phi) * jnp.sin(psi) * jnp.sin(theta))),
            -1 / self.m * (self.kt * dz - self.m * self.g + u1 * (jnp.cos(phi) * jnp.cos(theta))),
            dphi,
            dtheta,
            dpsi,
            -1 / self.Ixx * (self.kr * dphi - self.l * u2 - self.Iyy * dtheta * dpsi + self.Izz * dtheta * dpsi),
            -1 / self.Iyy * (-self.kr * dtheta + self.l * u3 - self.Ixx * dphi * dpsi + self.Izz * dphi * dpsi),
            -1 / self.Izz * (u4 - self.kr * self.r + self.Ixx * dphi * dtheta - self.Iyy * dphi * dtheta)
        ])

        return s + ds * dt
    
if __name__ == "__main__":
    testcopter = QuadcopterPlanar()

    t = np.linspace(0, 10, 101)

    s = np.zeros((101, 6))
    s[:, 1] = np.linspace(6 + testcopter.h, 0 + testcopter.h, 101)
    s[:, 4] = 0.1 * np.pi * np.sin(0.1 * t)

    sg = np.zeros((101, 6))
    sg[:, 4] = 0.1 * np.sin(t)
    sg[:, 0] = 0.3 * np.sin(t)

    print(s[:, 1])
    print(s[:, 4])
    print(sg[:, 4])

