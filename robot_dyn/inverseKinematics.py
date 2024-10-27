import pinocchio as pin
import numpy as np
import time
from numpy.linalg import pinv,inv,norm,svd,eig
from utils.tiago_loader import loadTiago
import matplotlib.pylab as plt
from utils.meshcat_viewer_wrapper import MeshcatVisualizer


robot = loadTiago()
viz = MeshcatVisualizer(robot)


viz.viewer.open()

q = pin.randomConfiguration(robot.model)
# vq in [-1, 1]
vq = np.random.rand(robot.model.nv) * 2 - 1
DT = 1e-3
qnext = pin.integrate(robot.model, q, vq * DT)

for t in range(1000):
    q = pin.integrate(robot.model, q, vq * DT)
    viz.display(q)
    time.sleep(DT / 10)

input('Press ENTER to exit.')