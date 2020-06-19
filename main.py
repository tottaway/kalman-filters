from rocket import Rocket
from plot import Plotter
import matplotlib.pyplot as plt
import numpy as np
import time
from pyqtgraph.Qt import QtCore, QtGui

if __name__ == "__main__":
    initial_state = np.array([0., 0., 0., 0., 1e-2, 0., 0., 0., 0., 0.])
    p = Plotter(0, 0, 0, -0.1, 0.1, np.array([0, 0, -9.8]))
    r = Rocket(*initial_state)
    def update():
        r.propagate_state()
        p.update(r.x, r.y, r.z, r.theta1, r.theta2, r.c)


    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(10)
    states = np.ones((100, 10))
    p.start()
    
    # labels = ['x', 'y', 'z', 'theta1', 'theta2', 'xdot', 'ydot', 'zdot',
    #           'theta1dot', 'theta2dot']
    # colors = ['r', 'g', 'b', 'c', 'k', 'r', 'g', 'b', 'c', 'k']
    # alphas = [1, 1, 1, 1, 1, 0.6, 0.6, 0.6, 0.6, 0.6]
    # x = np.linspace(0, 100, 100)
    # for label, data, color, alpha in zip(labels, states.T, colors, alphas):
    #     plt.plot(x, data, label=label, c=color, alpha=alpha)
    # plt.legend()
    # plt.show()

