import math
import time
import pinocchio as pin
import numpy as np
from numpy.random import rand
from numpy.linalg import inv, pinv, norm, eig, svd
import matplotlib.pylab as plt
import quadprog
from pinocchio.visualize import GepettoVisualizer

from robot_dyn.utils.robot_hand import RobotHand
from utils.meshcat_viewer_wrapper import MeshcatVisualizer


A = np.random.rand(5,5)*2-1
A = A @ A.T ### Make it positive symmetric
b = np.random.rand(5)

C = np.random.rand(10, 5)
d = np.random.rand(10)

[x,cost,_,niter,lag,iact] = quadprog.solve_qp(A,b,C.T,d)  # Notice that C.T is passed instead of C


robot = RobotHand()
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)

viz.viewer.open()

q = robot.q0.copy()

for i in range(500): # Put 1000 or 5000 if you want a longer move.
    for iq in range(3,robot.model.nq):
        q[iq] = -1+np.cos(i*1e-2*(1+iq/5))
    viz.display(q)
    time.sleep(2e-3)

q = robot.q0.copy()
vq = np.zeros(robot.model.nv)

M = pin.crba(robot.model, robot.data, q)
b = pin.nle(robot.model, robot.data, q, vq)

tauq = np.random.rand(robot.model.nv)
aq =  inv(M) @ (tauq - b)

print(norm(pin.rnea(robot.model, robot.data, q, vq, aq) - tauq))
dt = 2e-3
N_steps = 5000

# 自由落体运动 带摩擦力
q = robot.q0.copy()
viz.display(q)
vq = np.zeros(robot.model.nv)

for it in range(N_steps):
    t = it*dt

    # Retrieve the dynamics quantity at time t
    M = pin.crba(robot.model, robot.data, q)
    b = pin.nle(robot.model, robot.data, q, vq)

    # Compute the force that apply
    tauq = np.zeros(robot.model.nv) - 0.1 * vq

    # Use generalized PFD to calculate aq
    aq = inv(M) @ (tauq - b)

    # Double integration to update vq and q
    vq += aq * dt
    q = pin.integrate(robot.model, q, vq * dt)

    # Visualization
    if it % 20 == 0:
        viz.display(q)
        time.sleep(20*dt)




input('Press ENTER to exit.')



