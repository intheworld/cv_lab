import os
import time
import numpy as np
from pinocchio.visualize import MeshcatVisualizer
from scipy.optimize import fmin_bfgs,fmin_slsqp
from numpy.linalg import norm,inv,pinv,svd,eig
from utils.meshcat_viewer_wrapper import MeshcatVisualizer, planar, translation2d


viz = MeshcatVisualizer()
viz.viewer.open()

ballID = 'world/ball'
viz.addSphere(ballID,.2,[1,0,0,1])
cylID = 'world/cyl'
viz.addCylinder(cylID,length=1,radius=.1,color=[0,0,1,1])
boxID = 'world/box'
viz.addBox(boxID,[.5,.2,.4],[1,1,0,1])

viz.delete(ballID)

viz.applyConfiguration(cylID,[.1,.2,.3,1,0,0,0])
viz.applyConfiguration(boxID,planar(0.1, 0.2, np.pi / 3))
viz.applyConfiguration(cylID,planar(0.1, 0.2, 5*np.pi / 6))


# create new robot
viz.delete(ballID)
viz.delete(cylID)
viz.delete(boxID)

viz.addSphere('joint1',.1,[1,0,0,1])
viz.addSphere('joint2',.1,[1,0,0,1])
viz.addSphere('joint3',.1,[1,0,0,1])
viz.addCylinder('arm1',.75,.05,[.65,.65,.65,1])
viz.addCylinder('arm2',.75,.05,[.65,.65,.65,1])
viz.addSphere('target',.05,[0,.8,.1,1])

q = np.random.rand(2) * 2 * np.pi - np.pi

def display(q):
    """Display the robot in Gepetto Viewer. """
    assert (q.shape == (2,))
    c0 = np.cos(q[0])
    s0 = np.sin(q[0])
    c1 = np.cos(q[0] + q[1])
    s1 = np.sin(q[0] + q[1])
    viz.applyConfiguration('joint1',planar(0,           0,           0))
    viz.applyConfiguration('arm1'  ,planar(c0 / 2,      s0 / 2,      q[0]))
    viz.applyConfiguration('joint2',planar(c0,          s0,          q[0]))
    viz.applyConfiguration('arm2'  ,planar(c0 + c1 / 2, s0 + s1 / 2, q[0] + q[1]))
    viz.applyConfiguration('joint3',planar(c0 + c1,     s0 + s1,     q[0] + q[1]))

display(q)

# Optimize configuration

def endeffector(q):
    assert (q.shape == (2,))
    c0 = np.cos(q[0])
    s0 = np.sin(q[0])
    c1 = np.cos(q[0] + q[1])
    s1 = np.sin(q[0] + q[1])
    x, y = c0 + c1, s0 + s1
    return np.array([x, y])

print(endeffector(q))

target = np.array([.5, .5])
viz.applyConfiguration('target', translation2d(target[0], target[1]))

def cost(q):
    eff = endeffector(q)
    return norm(eff - target)**2

def callback(q):
    display(q)
    time.sleep(.5)


q0 = np.array([0.0, 0.0])
qopt_bfgs = fmin_bfgs(cost, q0, callback=callback)
print('\n *** Optimal configuration from BFGS = {} \n', qopt_bfgs)


# Optimize configration in cartesian coordinate

print('\n\n\n\n *** Optimize configration in cartesian coordinate')
x1, y1, th1, x2, y2, th2, x3, y3, th3 = q0 = np.zeros(9)

def endeffector_9(ps):
    assert (ps.shape == (9,))
    x1, y1, t1, x2, y2, t2, x3, y3, t3 = ps
    return np.array([x3, y3])

def display_9(q):
    """Display the robot in Viewer. """
    assert (q.shape == (9,))
    x1, y1, t1, x2, y2, t2, x3, y3, t3 = q
    viz.applyConfiguration('joint1',planar(x1,           y1,           t1))
    viz.applyConfiguration('arm1'  ,planar((x1 + x2) / 2, (y1 + y2) / 2, t1))
    viz.applyConfiguration('joint2',planar(x2,          y2,          t2))
    viz.applyConfiguration('arm2'  ,planar((x2 + x3) / 2, (y2 + y3) / 2, t2))
    viz.applyConfiguration('joint3',planar(x3, y3, t3))

def cost_9(ps):
    eff = endeffector_9(ps)
    return norm(eff - target)**2

qrand9 = np.random.rand(9)
display_9(qrand9)

def constraint_9(ps):
    assert (ps.shape == (9, ))
    x1, y1, t1, x2, y2, t2, x3, y3, t3 = ps
    res = np.zeros(7)
    res[0] = x1 - 0
    res[1] = y1 - 0
    res[2] = x1 + np.cos(t1) - x2
    res[3] = y1 + np.sin(t1) - y2
    res[4] = x2 + np.cos(t2) - x3
    res[5] = y2 + np.sin(t2) - y3
    res[6] = t3 - t2
    return res

print(cost_9(q0), constraint_9(q0))


def callback_9(ps):
    display_9(ps)
    time.sleep(.5)


def penalty(ps):
    return cost_9(ps) + 10 * norm(constraint_9(ps)) ** 2

qopt = fmin_bfgs(penalty, q0, callback=callback_9)

print('\n *** Optimal result. \n', qopt)

input('Press ENTER to exit.')