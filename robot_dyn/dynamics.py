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

# 逆动力学主要算法，RNEA (recursive Newton-Euler Algorithm)，作用是已知加速度求关节力矩
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

# 简单的pid控制
q = np.zeros(robot.model.nq)
viz.display(q)
vq = np.zeros(robot.model.nv)
# 仿真超参数
Kf = 0.1
Kp = 50.
Kv = 2 * np.sqrt(Kp)
alpha = 1
dt = 2e-3
N_steps = 5000

from utils.traj_ref import TrajRef

qdes = TrajRef(robot.q0, omega = np.array([0.,.1,1,1.5,2.5,-1,-1.5,-2.5,.1,.2,.3,.4,.5,.6]), amplitude = 1.5)

hq    = []   ### For storing the logs of measured trajectory q
hqdes = []   ### For storing the logs of desired trajectory qdes
for it in range(N_steps):
    t = it * dt

    M = pin.crba(robot.model, robot.data, q)
    b = pin.nle(robot.model, robot.data, q, vq)

    # Compute the force that apply
    tauq = - Kp * (q - qdes(t)) - Kv * (vq - qdes.velocity(t)) + alpha * qdes.acceleration(t)

    # Use generalized PFD to calculate aq
    aq = inv(M) @ (tauq - b)

    # Double integration to update vs and q
    vq += aq * dt
    q = pin.integrate(robot.model, q, vq * dt)

    # Visualization
    if it % 20 == 0:
        viz.display(q)
        time.sleep(20 * dt)

    # Log the history.
    hq.append(q.copy())
    hqdes.append(qdes.copy())

def plot_joint_profiles(i, hq, hqdes):
    plt.subplot(111)
    plt.plot([q[i] for q in hq])
    plt.plot([q[i] for q in hqdes])
    plt.ylabel(f'Joint {i}')
    plt.show()

plot_joint_profiles(11, hq, hqdes)

input('Press ENTER to exit.')



