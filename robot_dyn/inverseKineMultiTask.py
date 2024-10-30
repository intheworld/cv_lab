import pinocchio as pin
import numpy as np
import time
from numpy.linalg import pinv,inv,norm,svd,eig
from utils.tiago_loader import loadTiago
import matplotlib.pylab as plt
from utils.meshcat_viewer_wrapper import MeshcatVisualizer

# 多任务逆向动力学
robot = loadTiago(addGazeFrame=True)
viz = MeshcatVisualizer(robot)

IDX_TOOL = robot.model.getFrameId('frametool')
IDX_BASIS = robot.model.getFrameId('framebasis')

q0 = np.array([ 0.  ,  0.  ,  1.  ,  0.  ,  0.18,  1.37, -0.24, -0.98,  0.98,
                0.  ,  0.  ,  0.  ,  0.  , -0.13,  0.  ,  0.  ,  0.  ,  0.  ])
DT = 1e-2

IDX_GAZE = robot.model.getFrameId('framegaze')

# Add a small ball as a visual target to be reached by the robot
ball = np.array([ 1.2,0.5,1.1 ])
viz.addSphere('ball', .05, [ .8,.1,.5, .8])
viz.applyConfiguration('ball', list(ball)+[0,0,0,1])

# Add the box again
oMgoal = pin.SE3(pin.Quaternion(-0.5, 0.58, -0.39, 0.52).normalized().matrix(), np.array([1.2, .4, .7]))
viz.addBox('goal', [.1,.1,.1], [ .1,.1,.5, .6])
viz.applyConfiguration('goal', oMgoal)

viz.display(q0)
viz.viewer.open()

## gaze control
q = q0.copy()
herr = []  # Log the value of the error between gaze and ball.
# Loop on an inverse kinematics for 200 iterations.
for i in range(500):  # Integrate over 2 second of robot life

    # Run the algorithms that outputs values in robot.data
    pin.framesForwardKinematics(robot.model, robot.data, q)
    pin.computeJointJacobians(robot.model, robot.data, q)

    # Placement from world frame o to frame f oMgaze
    oMgaze = robot.data.oMf[IDX_GAZE]

    # 6D jacobian in local frame
    o_Jgaze3 = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_GAZE, pin.LOCAL_WORLD_ALIGNED)[:3, :]

    # vector from gaze to ball, in world frame
    o_GazeBall = oMgaze.translation - ball

    vq = -pinv(o_Jgaze3) @ o_GazeBall

    q = pin.integrate(robot.model, q, vq * DT)
    viz.display(q)
    time.sleep(1e-3)

    herr.append(o_GazeBall)

# 多任务
q = q0.copy()
herr = [] # Log the value of the error between tool and goal.
herr2 = [] # Log the value of the error between gaze and ball.
# Loop on an inverse kinematics for 200 iterations.
for i in range(500):  # Integrate over 2 second of robot life

    # Run the algorithms that outputs values in robot.data
    pin.framesForwardKinematics(robot.model,robot.data,q)
    pin.computeJointJacobians(robot.model,robot.data,q)

    # Tool task
    oMtool = robot.data.oMf[IDX_TOOL]
    tool_Jtool = pin.computeFrameJacobian(robot.model,robot.data,q,IDX_TOOL,pin.LOCAL)
    tool_nu = pin.log(oMtool.inverse()*oMgoal).vector

    # Gaze task
    oMgaze = robot.data.oMf[IDX_GAZE]
    o_Jgaze3 = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_GAZE,pin.LOCAL_WORLD_ALIGNED)[:3,:]
    o_GazeBall = oMgaze.translation-ball

    vq = pinv(tool_Jtool) @ tool_nu
    Ptool = np.eye(robot.nv)-pinv(tool_Jtool) @ tool_Jtool
    vq += Ptool @ pinv(o_Jgaze3 @ Ptool) @ (-o_GazeBall - o_Jgaze3 @ vq)

    q = pin.integrate(robot.model,q, vq * DT)
    viz.display(q)
    time.sleep(1e-3)

    herr.append(tool_nu)
    herr2.append(o_GazeBall)

plt.subplot(311)
plt.plot([ e[:3] for e in herr])
plt.xlabel('control cycle (iter)')
plt.ylabel('error (m)')
plt.subplot(312)
plt.plot([ e[3:] for e in herr])
plt.xlabel('control cycle (iter)')
plt.ylabel('error (rad)')
plt.subplot(313)
plt.plot([ e for e in herr2])
plt.xlabel('control cycle (iter)')
plt.ylabel('error (rad)')

input('Press ENTER to exit.')