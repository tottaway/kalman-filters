import sys

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np

class Plotter():
    def __init__(self, x, y, z, theta1, theta2, thrust_vec):
        self.app = QtGui.QApplication([])
        self.w = gl.GLViewWidget()
        self.w.show()
        self.w.setWindowTitle('pyqtgraph example: GLMeshItem')
        self.w.setCameraPosition(distance=40)
        self.w.pan(200, 200, 0)
        self.w.orbit(0, -40)

        self.g = gl.GLGridItem()
        self.g.scale(2,2,1)
        self.w.addItem(self.g)

        heading = np.array([np.cos(theta1) * np.sin(theta2),
                            np.sin(theta1) * np.sin(theta2),
                            np.cos(theta2)])
        heading = heading / np.linalg.norm(heading)
        position = np.array([x, y, z])

        z_unit_vector = np.array([0, 0, 1])
        heading_z_cross = np.cross(z_unit_vector, heading)
        heading_angle = np.degrees(np.arccos(heading[2]))

        # main body
        self.main_body_length = 7.5
        main_body_data = gl.MeshData.cylinder(rows=10, cols=20, radius=[1.5, 1.5], length=self.main_body_length)
        colors = np.ones((main_body_data.faceCount(), 4), dtype=float)
        colors[::2,0] = 0
        colors[:,1] = np.linspace(0, 1, colors.shape[0])
        main_body_data.setFaceColors(colors)
        self.main_body = gl.GLMeshItem(meshdata=main_body_data, smooth=True, drawEdges=False,
                                  shader='balloon')

        self.main_body.rotate(heading_angle, *heading_z_cross)
        self.main_body.translate(*position)

        # top cone
        self.top_cone_length = 2.5
        top_cone_data = gl.MeshData.cylinder(rows=10, cols=20, radius=[1.5, 0], length=self.top_cone_length)
        colors = np.ones((top_cone_data.faceCount(), 4), dtype=float)
        colors[::2,0] = 0
        colors[:,1] = np.linspace(0, 1, colors.shape[0])
        top_cone_data.setFaceColors(colors)
        self.top_cone = gl.GLMeshItem(meshdata=top_cone_data, smooth=True, drawEdges=False,
                                  shader='balloon')

        self.top_cone.rotate(heading_angle, *heading_z_cross)
        self.top_cone.translate(*position)
        self.top_cone.translate(*(self.main_body_length* heading))

        # bottom_cone
        self.bottom_cone_length = 0.5
        bottom_cone_data = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.5, 1.5], length=self.bottom_cone_length)
        colors = np.ones((bottom_cone_data.faceCount(), 4), dtype=float)
        colors[::2,0] = 0
        colors[:,1] = np.linspace(0, 1, colors.shape[0])
        bottom_cone_data.setFaceColors(colors)
        self.bottom_cone = gl.GLMeshItem(meshdata=bottom_cone_data, smooth=True, drawEdges=False,
                                  shader='balloon')

        self.bottom_cone.rotate(heading_angle, *heading_z_cross)
        self.bottom_cone.translate(*position)
        self.bottom_cone.translate(*(-0.5 * heading))


        thrust_mag = np.linalg.norm(thrust_vec)
        thrust_data = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.5, 0], length=1)
        colors = np.ones((thrust_data.faceCount(), 4), dtype=float)
        colors[::2,0] = 0
        colors[:,1] = np.linspace(0, 1, colors.shape[0])
        thrust_data.setFaceColors(colors)
        self.thrust = gl.GLMeshItem(meshdata=thrust_data, smooth=True, drawEdges=False,
                                  shader='balloon')

        z_unit_vector = np.array([0, 0, 1])
        thrust_unit_vector = thrust_vec / thrust_mag
        cross = np.cross(z_unit_vector, thrust_unit_vector)
        if np.linalg.norm(cross) == 0:
            self.thrust.rotate(180, 1, 0, 0)
        else:
            thrust_angle = np.degrees(np.arccos(thrust_unit_vector[2]))
            self.thrust.rotate(thrust_angle, *cross)

        self.thrust.scale(1, 1, thrust_mag/5)
        self.thrust.translate(*position)
        self.thrust.translate(*(-0.25 * heading))
            
        self.w.addItem(self.main_body)
        self.w.addItem(self.thrust)
        self.w.addItem(self.bottom_cone)
        self.w.addItem(self.top_cone)


    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def update(self, x, y, z, theta1, theta2, thrust_vec):
        self.main_body.resetTransform()
        self.top_cone.resetTransform()
        self.bottom_cone.resetTransform()
        self.thrust.resetTransform()

        position = np.array([x, y, z])
        heading = np.array([np.cos(theta1) * np.sin(theta2),
                            np.sin(theta1) * np.sin(theta2),
                            np.cos(theta2)])
        heading_unit_vec = heading / np.linalg.norm(heading)

        self.main_body.rotate(np.degrees(theta2), 0, 1, 0)
        self.top_cone.rotate(np.degrees(theta2), 0, 1, 0)
        self.bottom_cone.rotate(np.degrees(theta2), 0, 1, 0)

        self.main_body.rotate(np.degrees(theta1), 0, 0, 1)
        self.top_cone.rotate(np.degrees(theta1), 0, 0, 1)
        self.bottom_cone.rotate(np.degrees(theta1), 0, 0, 1)

        self.top_cone.translate(*(heading_unit_vec * self.main_body_length))
        self.bottom_cone.translate(*(heading_unit_vec * -0.5))

        self.main_body.translate(*position)
        self.top_cone.translate(*position)
        self.bottom_cone.translate(*position)



        thrust_mag = np.linalg.norm(thrust_vec)
        thrust_length = thrust_mag / 5
        thrust_unit_vector = thrust_vec / thrust_mag
        thrust_z_cross = np.cross(np.array([0, 0, 1]), thrust_unit_vector)
        if np.linalg.norm(thrust_z_cross) == 0:
            self.thrust.rotate(180, 1, 0, 0)
        else:
            thrust_angle = np.degrees(np.arccos(thrust_unit_vector[2]))
            self.thrust.rotate(thrust_angle, *thrust_z_cross)

        self.thrust.scale(1, 1, thrust_length)
        self.thrust.translate(*position)
        self.thrust.translate(*(-0.25 * heading))

