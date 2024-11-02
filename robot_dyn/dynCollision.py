import math
import time
import pinocchio as pin
import numpy as np
from numpy.random import rand
from numpy.linalg import inv, pinv, norm, eig, svd
import matplotlib.pylab as plt
import quadprog
from pinocchio.visualize import GepettoVisualizer

from utils.robot_hand import RobotHand
from utils.traj_ref import TrajRef
from utils.meshcat_viewer_wrapper import MeshcatVisualizer
from utils.collision_wrapper import CollisionWrapper

robot = RobotHand()
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)

viz.viewer.open()

q = robot.q0.copy()

colwrap = CollisionWrapper(robot)
colwrap.computeCollisions(q)
collisions = colwrap.getCollisionList()

dist = colwrap.getCollisionDistances(collisions)
J = colwrap.getCollisionJacobian(collisions)

q = np.zeros(robot.model.nq)
viz.display(q)
vq = np.zeros(robot.model.nv)

# Hyperparameters for the control and the simu
Kf = 0.1
Kp = 50.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)
alpha = 1.

dt = 1e-3             # simulation timestep
N_steps = 10000

Kp_c = 100.
Kv_c = 2 * np.sqrt(Kp_c)

qdes = TrajRef(robot.q0, omega = np.array([0.,.1,1,1.5,2.5,-1,-1.5,-2.5,.1,.2,.3,.4,.5,.6]), amplitude = 1.5)

hq_c    = []   ### For storing the logs of measured trajectory q
hqdes_c = []   ### For storing the logs of desired trajectory qdes
tracked_collisions_id = set()  # Track contact

for it in range(N_steps):
    t = it * dt
    # CRBA（Composite Rigid Body Algorithm）算法，在pin这里的含义是惯性矩阵
    M = pin.crba(robot.model, robot.data, q)
    b = pin.nle(robot.model, robot.data, q, vq)

    # Compute the force that apply
    tauq_control = - Kp * (q - qdes(t)) - Kv * (vq - qdes.velocity(t)) + alpha * qdes.acceleration(t)

    tauq_friction = - Kf * vq

    # Use generalized PFD to calculate aq
    aq0 = inv(M) @ (tauq_friction + tauq_control - b)

    # 检查冲突
    colwrap.computeCollisions(q, vq)
    raw_collisions = colwrap.getCollisionList()
    raw_dist = colwrap.getCollisionDistances(raw_collisions)

    # 只关注重要的冲突
    collisions = [c for c, d in zip(raw_collisions, raw_dist) if d <= -1e-4]

    if len(collisions) <= 0:
        aq = aq0
        tracked_collisions_id = set()
    else:
        dists = colwrap.getCollisionDistances(collisions)
        J = colwrap.getCollisionJacobian(collisions)
        JdotQdot = colwrap.getCollisionJdotQdot(collisions)

        collisions_id = [col[0] for col in collisions]
        new_collisions_pyidx = [
            pyidx
            for pyidx, col_id in enumerate(collisions_id)
            if col_id not in tracked_collisions_id
        ]
        tracked_collisions_id = set(collisions_id)

        if new_collisions_pyidx:
            J_proj = np.stack([J[i] for i in new_collisions_pyidx], axis = 0)
            vq -= (pinv(J_proj) @ J_proj) @ vq

        # 这里依据了高斯最小约束原理，也是一种变分发方法, todo: 熟悉这一块东西
        A = M
        b = M @ aq0
        C = J
        d = - JdotQdot - Kp_c * dists - Kv_c * J @ vq

        [aq, cost, _, niter, lag, iact] = quadprog.solve_qp(A, b, C.T, d)

    # Double integration to update vs and q
    vq += aq * dt
    q = pin.integrate(robot.model, q, vq * dt)

    # Visualization
    if it % 50 == 0:
        viz.display(q)
        time.sleep(20 * dt)

    # Log the history.
    hq_c.append(q.copy())
    hqdes_c.append(qdes.copy())

def plot_joint_profiles(i, hq, hqdes):
    plt.subplot(111)
    plt.plot([q[i] for q in hq])
    plt.plot([q[i] for q in hqdes])
    plt.ylabel(f'Joint {i}')
    plt.show()

plot_joint_profiles(11, hq_c, hqdes_c)

input('Press ENTER to exit.')