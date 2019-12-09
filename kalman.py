import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint

###############################################################################
# Set up state variables for kalman filter as well as a ground truth state ####
###############################################################################
true_state = np.array([
    [0], # angle
    [1]  # angular velocity
], dtype=float)

state = np.array([
    [0], # angle
    [0]  # angular velocity
], dtype=float)

last_state = np.array([
    [0], # angle
    [0]  # angular velocity
], dtype=float)

# linear approximation for a pendulum system
A = np.array([
     [0, 1],
     [-1, 0]
], dtype=float)

# State model for actions
# currently our system is un-accuated, therefore we leave B as the identity
# knowing that it will be canceled
B = np.array([
    [1, 0],
    [0, 1]
], dtype=float)

# actions (currently unused)
u = np.array([
    [0],
    [0]
], dtype=float)

# Represent how confident filter is
P = np.array([
    [1, 0],
    [0, 1]
], dtype=float)

# covariance matrix for action
# theoretically determined through experiments
Q = np.array([
    [1, 0],
    [0, 1]
], dtype=float)

# Matrix used in intermediate calculations
K = np.array([
    [1, 0],
    [0, 1]
], dtype=float)

# model of sensors (not entirely sure how this affects things)
H = np.array([
    [1, 0],
    [0, 1]
], dtype=float)

# covariance matrix for sensors
R = np.array([
    [1, 0],
    [0, 1]
], dtype=float)

# measurements recieved from sensors
z = np.array([
    [1],
    [0]
], dtype=float)

###############################################################################
# Helper Functions ############################################################
###############################################################################

def noise():
    """returns some amount of noise pulled from the normal distribution"""
    sigma = 0.5
    mean = 0
    return np.random.rormal(mean, sigma, (2, 1))

def update_state(dt):
    """Propagates system based on model"""
    global true_state
    tmp_x = true_state[0][0]
    true_state[0][0] += true_state[1] * dt
    true_state[1][0] += -1*np.sin(tmp_x) * dt
    # damping
    true_state[1][0] += -0.01 * true_state[1][0]

def read_sensors():
    return true_state + noise()

###############################################################################
# Kalman Filter ###############################################################
###############################################################################

def kalman(dt):
    """ Use the kalman filter to predict the next state """
    global state, last_state
    last_state = state
    state = read_sensors()

    # prediction
    x = last_state + (A @ last_state * dt) + u + (B @ u * dt)
    p = A @ P @ A.T + Q

    # Update Equations
    K = p @ H.T @ np.linalg.pinv(H @ p @ H.T + R)
    state = x + K @ (state - H @ x)
    p = (np.eye(2) - K@H) @ p

###############################################################################
# Run Simulation ##############################################################
###############################################################################

def run():
    """
    run a simulation of the swinging pendulum and plot the resulting true
    states, kalman filter predictions, and sensor readings
    """
    total_time = 30
    dt = 0.01
    t = np.linspace(0, total_time, int(total_time/dt))
    true_xs = np.zeros(int(total_time/dt))
    kalman_xs = np.zeros(int(total_time/dt))
    sensor_xs = np.zeros(int(total_time/dt))
    

    for i, curr_t in tqdm(enumerate(t)):
        update_state(dt)        
        true_xs[i] = true_state[0][0]
        kalman(dt)
        kalman_xs[i] = state[0][0]

        # currently I'm not displaying the same reading the filter is seeing
        # however, they are still helpful for seeing how much better the filter
        # is
        sensor_xs[i] = read_sensors()[0][0] 
    
    plt.plot(t, true_xs)
    plt.plot(t, kalman_xs)
    plt.plot(t, sensor_xs, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    P = np.array([
        [0.01, 0],
        [0, 0.01]
    ], dtype=float)
    R = np.array([
        [40, 0],
        [0, 40]
    ], dtype=float)
    run()

    # reset state and run again with different parameters
    true_state = np.array([
        [0],
        [1] 
    ], dtype=float)
    P = np.array([
        [0.01, 0],
        [0, 0.001]
    ], dtype=float)
    R = np.array([
        [20, 0],
        [0, 20]
    ], dtype=float)
    run()
