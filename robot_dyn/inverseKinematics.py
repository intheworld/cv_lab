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

IDX_TOOL = robot.model.getFrameId('frametool')
IDX_BASIS = robot.model.getFrameId('framebasis')

print(robot.model.frames[IDX_TOOL])

pin.framesForwardKinematics(robot.model, robot.data, q)

oMtool = robot.data.oMf[IDX_TOOL]
oMbasis = robot.data.oMf[IDX_BASIS]

print("Tool placement:", oMtool)
print("Basis placement:", oMbasis)

### Computing Jacobian

Jtool = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_TOOL)
print("Jtool shape = ", Jtool.shape)
print('Jtool = ', Jtool)

input('Press ENTER to exit.')