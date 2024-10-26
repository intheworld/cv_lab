import time
import numpy as np
from numpy.linalg import norm
import pinocchio as pin
import example_robot_data as robex
from scipy.optimize import fmin_bfgs
from utils.meshcat_viewer_wrapper import MeshcatVisualizer, colors

# Optimizing in the quaternion space
robot = robex.load('solo12')
viz = MeshcatVisualizer(robot)
viz.viewer.open()

robot.feetIndexes = [robot.model.getFrameId(frameName) for frameName in ['HR_FOOT', 'HL_FOOT', 'FR_FOOT', 'FL_FOOT']]

# --- Add box to represent target
theColors = ['red', 'blue', 'green', 'magenta']
for color in theColors:
    viz.addSphere("world/%s" % color, .05, color)
    viz.addSphere("world/%s_des" % color, .05, color)

#
# OPTIM 6D #########################################################
#

targets = [
    np.array([-0.7, -0.2, 1.2]),
    np.array([-0.3, 0.5, 0.8]),
    np.array([0.3, 0.1, -0.1]),
    np.array([0.9, 0.9, 0.5])
]
for i in range(4):
    targets[i][2] += 1


def cost(q):
    '''Compute score from a configuration'''
    cost = 0.
    for i in range(4):
        p_i = robot.framePlacement(q, robot.feetIndexes[i]).translation
        cost += norm(p_i - targets[i])**2
    return cost


def callback(q):
    viz.applyConfiguration('world/box', Mtarget)

    for i in range(4):
        p_i = robot.framePlacement(q, robot.feetIndexes[i])
        viz.applyConfiguration('world/%s' % theColors[i], p_i)
        viz.applyConfiguration('world/%s_des' % theColors[i], list(targets[i]) + [1, 0, 0, 0])

    viz.display(q)
    time.sleep(1e-1)


Mtarget = pin.SE3(pin.utils.rotate('x', 3.14 / 4), np.array([0.5, 0.1, 0.2]))  # x,y,z
qopt = fmin_bfgs(cost, robot.q0, callback=callback)

input('Press ENTER to exit.')