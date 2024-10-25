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
    '''Display the robot in Gepetto Viewer. '''
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


input('Press ENTER to exit.')