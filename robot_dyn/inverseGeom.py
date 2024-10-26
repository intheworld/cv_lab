import time
import math
import numpy as np
from numpy.linalg import norm
import pinocchio as pin
import example_robot_data as robex
from scipy.optimize import fmin_bfgs
from utils.meshcat_viewer_wrapper import MeshcatVisualizer, colors


robot = robex.load('ur5')
print(robot.model)

robot.index('wrist_3_joint')
for i, n in enumerate(robot.model.names):
    print(i, n)

for f in robot.model.frames:
    print(f.name, 'attached to joint #', f.parentJoint)

a = robot.placement(robot.q0, 6)  # Placement of the end effector joint.
b = robot.framePlacement(robot.q0, 22)  # Placement of the end effector tip.

tool_axis = b.rotation[:, 2]  # Axis of the tool

NQ = robot.model.nq
NV = robot.model.nv  # for this simple robot, NV == NQ

viz = MeshcatVisualizer(robot)

viz.viewer.open()

q = np.array([-1., -1.5, 2.1, -.5, -.5, 0])

viz.display(q)

# Add a red box in the viewer
ballID = "world/ball"
radius = 0.1
viz.addSphere(ballID, radius, colors.red)

# Place the ball at the position ( 0.5, 0.1, 0.2 )
# The viewer expect position and rotation, append the identity quaternion
o_ball = np.array([0.5, 0.1, 0.2])
q_ball = o_ball.tolist() +  [1, 0, 0, 0]
viz.applyConfiguration(ballID, q_ball)

m = robot.framePlacement(q, 22)  # SE(3) element frame of the tip
p = m.translation  # Position of the tip
ez = m.rotation[:, 2]  # Direction of the tip

target = np.array(o_ball)  # x,y,z

def cost(q):
    '''Compute score from a configuration'''
    m = robot.framePlacement(q, 22)
    p = m.translation
    offset = m.rotation[:, 2] * radius
    return norm(p +  offset - target)**2


def callback(q):
    viz.display(q)
    time.sleep(1e-2)

q_touch = fmin_bfgs(cost, robot.q0, callback=callback)

q = q_touch.copy()
vq = np.array([2., 0, 0, 4., 0, 0])
idx = 6

oMend = robot.placement(q, idx)
o_end = oMend.translation  # Position of end-eff express in world frame
o_ball = q_ball[:3]  # Position of ball express in world frame
o_end_ball = o_ball - o_end  # Relative position of ball center wrt end effector position, express in world frame
end_ball = oMend.rotation.T @ o_end_ball  # Position of ball wrt eff in local coordinate

for i in range(200):
    # Chose new configuration of the robot
    q += vq / 40
    q[2] = 1.71 + math.sin(i * 0.05) / 2

    # Gets the new position of the ball
    oMend = robot.placement(q, idx)
    o_ball = oMend * end_ball  # Apply oMend to the relative placement of ball

    # Display new configuration for robot and ball
    viz.applyConfiguration(ballID, o_ball.tolist() + [1, 0, 0, 0])
    viz.display(q)
    time.sleep(1e-2)


# Add red box in the viewer
boxID = "world/box"
try:
    viz.delete(ballID)
except:
    pass

viz.addBox(boxID, [0.1, 0.2, 0.1], colors.magenta)

oMbox = pin.SE3(np.eye(3), np.array([0.5, 0.1, 0.2]))

viz.applyConfiguration(boxID, oMbox)

boxMtarget = pin.SE3(pin.utils.rotate('x', -np.pi / 2), np.array([0., -0.1, 0.]))
oMtarget = oMbox * boxMtarget


def cost(q):
    """Compute score from a configuration"""
    oMtip = robot.framePlacement(q, 22)
    # Align tip placement and facet placement
    return norm(pin.log(oMtip.inverse() * oMtarget).vector)


def callback(q):
    viz.display(q)
    time.sleep(1e-2)


qopt = fmin_bfgs(cost, robot.q0, callback=callback)
print('The robot finally reached effector placement at\n', robot.placement(qopt, 6))

q = qopt.copy()
vq = np.array([2., 0, 0, 4., 0, 0])
idx = 6

oMend = robot.placement(q, idx)
endMbox = oMend.inverse() * oMbox  # Placement of the box wrt end effector

for i in range(100):
    # Chose new configuration of the robot
    q += vq / 40
    q[2] = 1.71 + math.sin(i * 0.05) / 2

    oMend = robot.placement(q, idx)
    oMbox = oMend * endMbox

    # Display new configuration for robot and box
    viz.applyConfiguration(boxID, oMbox)
    viz.display(q)
    time.sleep(1e-2)

input('Press ENTER to exit.')