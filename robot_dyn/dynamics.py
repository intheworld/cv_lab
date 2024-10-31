import math
import time
import pinocchio as pin
import numpy as np
from numpy.random import rand
from numpy.linalg import inv, pinv, norm, eig, svd
import matplotlib.pylab as plt
import quadprog
from pinocchio.visualize import GepettoVisualizer

from utils.meshcat_viewer_wrapper import MeshcatVisualizer
from utils.robot_hand import Robot


A = np.random.rand(5,5)*2-1
A = A @ A.T ### Make it positive symmetric
b = np.random.rand(5)

C = np.random.rand(10, 5)
d = np.random.rand(10)

[x,cost,_,niter,lag,iact] = quadprog.solve_qp(A,b,C.T,d)  # Notice that C.T is passed instead of C


robot = Robot()
robot.display(robot.q0)


input('Press ENTER to exit.')



