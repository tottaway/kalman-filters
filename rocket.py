import numpy as np
import sys

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from scipy.spatial.transform import Rotation

class Rocket():
    def __init__(self, x, y, z, theta1, theta2, xdot, ydot, zdot, theta1dot,
                 theta2dot, timestep=0.1):
        self.x = x
        self.y = y
        self.z = z
        self.theta1 = theta1
        self.theta2 = theta2
        if self.theta2 == 0:
            self.theta2 = 1e-10
        self.xdot = xdot
        self.ydot = ydot
        self.zdot = zdot
        self.theta1dot = theta1dot
        self.theta2dot = theta2dot
        self.dt = timestep

        self.L = -3. # distance from CG of rocket ot the motor
        self.Dv = 1. # linear drag on the rocket
        self.Dw = 3 # rotational drag on the rocket
        self.I = 1. # roatational inertia of the rocket
        self.g = 9.81 # acceleration due to gravity
        self.m = 1. # mass


        # distance from the CG of the rocket to the center of aerodynamic
        # pressure
        self.l = -1 

        self.c = np.array([0, 0, 0])


    def _control(self):
        thrust_vector = -self.heading / np.linalg.norm(self.heading)
        return 1.5 * thrust_vector * (self.g * self.m)
        # return np.array([0, 0, 0])


    def _angle_between_vectors(self, vector1, vector2):
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 == 0 or norm2 == 0:
            return 0
        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = vector2 / np.linalg.norm(vector2)

        return np.arccos(np.dot(vector1, vector2))

    @property
    def thrust_mag(self):
        return np.linalg.norm(self.c)

    @property
    def v(self):
        return np.array([self.xdot, self.ydot, self.zdot])

    @property
    def v_mag(self):
        return np.linalg.norm(self.v)

    @property
    def v2(self):
        return self.v_mag ** 2

    @property
    def heading(self):
        """returns a vector pointing in the same direction as the rocket"""
        return np.array([np.cos(self.theta1) * np.sin(self.theta2),
                         np.sin(self.theta1) * np.sin(self.theta2),
                         np.cos(self.theta2)])
    @property
    def phi1(self):
        """return xy angle of velocity"""
        xy_velocity = self.v.copy()
        xy_velocity[2] = 0
        return self._angle_between_vectors(xy_velocity, np.array([1, 0, 0]))

    @property
    def phi2(self):
        """returns the angle between velocity and z axis"""
        return self._angle_between_vectors(self.v, np.array([0, 0, 1]))

    def _magnitude_of_projection(self, vector1, vector2):
        """takes in two vectors, and returns the magnitude of the projection of
        the first onto the second
        """
        # normalize vector2
        vector2 /= np.linalg.norm(vector2)
        return np.dot(vector1, vector2)

    def _calculate_forces_along_vector(self, unit_vector, forces):
        total_force = 0
        for force in forces:
            total_force += np.dot(force, unit_vector)
        return total_force

    def _calculate_xddot(self, linear_drag, thrust):
        forces = [linear_drag, thrust]
        unit_vector = np.array([1, 0, 0])
        force = self._calculate_forces_along_vector(unit_vector, forces)
        return (1/self.m) * force

    def _calculate_yddot(self, linear_drag, thrust):
        forces = [linear_drag, thrust]
        unit_vector = np.array([0, 1, 0])
        force = self._calculate_forces_along_vector(unit_vector, forces)
        return (1/self.m) * force

    def _calculate_zddot(self, linear_drag, thrust):
        forces = [linear_drag, thrust]
        unit_vector = np.array([0, 0, 1])
        force = self._calculate_forces_along_vector(unit_vector, forces)
        return (1/self.m) * force - self.g

    def _calculate_theta1ddot(self):
        # TODO: figure out rotation drag values
        # looking for torques in the xy plane
        xy_heading = self.heading.copy()
        xy_heading[2] = 0
        # torque_direction is the direction along which a force is all torque
        torque_direction = np.cross(xy_heading, np.array([0, 0, 1]))
        # normalize torque_direction
        torque_direction /= np.linalg.norm(torque_direction)

        # calculate theta1 torque due to thrust
        thrust_force = np.dot(self.c, torque_direction)
        effective_arm_len = np.sin(self.theta2) * self.L
        thrust_torque = thrust_force * effective_arm_len

        # calculate theta1 torque due to air pressure
        projected_velocity = self._magnitude_of_projection(self.v, torque_direction)
        air_pressure_force = projected_velocity**2 * self.Dw
        effective_arm_len = self.l * np.sin(self.theta1 - self.phi1)
        air_pressure_torque = air_pressure_force * effective_arm_len

        # rotational_drag
        drag = self.Dw * self.theta1dot * np.abs(self.theta1dot)

        return (1 / self.I) * (thrust_torque + air_pressure_torque - drag)

    def _calculate_theta2ddot(self):
        # TODO: figure out rotation drag values
        # looking for torques in the plane defined by the z-axis and heading
        # perpendicular to heading
        normal = np.cross(self.heading, np.array([0., 0., 1.]))
        torque_direction = np.cross(self.heading, normal)
        torque_direction /= np.linalg.norm(torque_direction)

        # calculate theta1 torque due to thrust
        thrust_force = self._magnitude_of_projection(self.c, torque_direction)
        effective_arm_len = self.L
        thrust_torque = thrust_force * effective_arm_len

        # calculate theta1 torque due to air pressure
        projected_velocity = self._magnitude_of_projection(self.v, torque_direction)
        air_pressure_force = projected_velocity**2 * self.Dw
        effective_arm_len = self.l * np.sin(self.theta2 - self.phi2)
        air_pressure_torque = air_pressure_force * effective_arm_len

        # rotational_drag
        drag = self.Dw * self.theta2dot * np.abs(self.theta2dot)

        return (1 / self.I) * (thrust_torque + air_pressure_torque - drag)

         
    def propagate_state(self):
        """
        propagate_state is a function which advances the state one time step
        """
        self.c = self._control() # give a thrust vector in cartesian coordinates
        thrust = -self.c
   
        velocity_angle_offset = self._angle_between_vectors(self.v, self.heading)
        linear_drag_mag = np.abs(self.v2*np.sin(velocity_angle_offset)*self.Dv)
        linear_drag = -linear_drag_mag * self.v

        xddot = self._calculate_xddot(linear_drag, thrust)
        yddot = self._calculate_yddot(linear_drag, thrust)
        zddot = self._calculate_zddot(linear_drag, thrust)


        theta1ddot = self._calculate_theta1ddot()
        theta2ddot = self._calculate_theta2ddot()

        # print("control %s" % self.c)
        # print("heading %s" % self.heading)
        # print("x %s" % self.x)
        # print("y %s" % self.y)
        # print("z %s" % self.z)
        # print("v %s" % self.v)
        # print("linear_drag %s" % linear_drag)
        # print("xdot %s" % self.xdot)
        # print("xddot %s" % xddot)
        # print("theta1 %s" % self.theta1)
        # print("theta2 %s" % self.theta2)
        # print("thetadot1 %s" % self.theta1dot)
        print("theta2dot %s" % self.theta2dot)
        # print("theta1ddot %s" % theta1ddot)
        print("theta2ddot %s" % theta2ddot)
        # print("phi1 %s" % self.phi1)
        # print("zdot %s" % self.zdot)
        # print("zddot %s" % zddot)
        print("")

        self.x += self.dt * self.xdot
        self.y += self.dt * self.ydot
        self.z += self.dt * self.zdot
        self.theta1 += self.dt * self.theta1dot
        self.theta2 += self.dt * self.theta2dot
        self.xdot += self.dt * xddot
        self.ydot += self.dt * yddot
        self.zdot += self.dt * zddot
        self.theta1dot += self.dt * theta1ddot
        self.theta2dot += self.dt * theta2ddot

        self.theta1 = self.theta1 % (2 * np.pi)
        self.theta2 = self.theta2 % (2 * np.pi)
        # avoid undefined behavior
        if self.theta2 == 0:
            self.theta2 = 1e-10

