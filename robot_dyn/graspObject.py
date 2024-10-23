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


input('Press ENTER to exit.')