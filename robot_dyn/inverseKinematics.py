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

### 线性速度的Jacobian
Jtool3 = Jtool[:3, :]
vtool = Jtool3 @ vq

tool_vtool = vtool
o_vtool = oMtool.rotation @ vtool

tool_Jtool = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_TOOL, pin.LOCAL)
tool_Jtool3 = tool_Jtool[:3,:]
o_Jtool3 = oMtool.rotation @ tool_Jtool3

EPS = 1e-4
u = (2 * np.random.rand(robot.model.nv) - 1)
dq = u * EPS

q2 = pin.integrate(robot.model, q, dq)

# tool position for q
pin.framesForwardKinematics(robot.model, robot.data, q)
o_M_tool = robot.data.oMf[IDX_TOOL].copy()
o_T_tool = o_M_tool.translation


# tool position for q + dq
pin.framesForwardKinematics(robot.model, robot.data, q2)
o_M_tool2 = robot.data.oMf[IDX_TOOL].copy()
o_T_tool2 = o_M_tool2.translation

print("Full Jacobian in the tool frame:")
print(f"With a Jac: {tool_Jtool @ u}")
print(f"With a log: {pin.log(o_M_tool.inverse() * o_M_tool2).vector / EPS}")

print("\nOrigin velocity in the world frame:")
print(f"With a Jac: {o_Jtool3 @ u}")
print(f"With a log: {o_M_tool.rotation @ pin.log(o_M_tool.inverse() * o_M_tool2).linear / EPS}")
print(f"With fdiff: {(o_T_tool2 - o_T_tool) / EPS}")


# o_w_tool and tool_w_tool
tool_Jtool = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_TOOL, pin.LOCAL)
o_Jtool = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_TOOL, pin.WORLD)
# the following expressions are equal.
print(f"With spatial Jacobian:        {o_Jtool @ u}")
print(f"With adjoint * body Jacobian: {oMtool.action @ (tool_Jtool @ u)}")

# 0_v_tool and tool_w_tool
o_Jtool3 = oMtool.rotation @ tool_Jtool[:3, :]
new_o_Jtool3 = pin.computeFrameJacobian(robot.model,robot.data, q, IDX_TOOL, pin.LOCAL_WORLD_ALIGNED)[:3,:]
# TODO: 为什么下面的两个表达式有些偏差？
print(f"o_Jtool3 from body Jacobian:  {o_Jtool3 @ u}")
print(f"Local-world-aligned Jacobian: {new_o_Jtool3 @ u}")


### 逆向运动学 3D
# 创建目标，四元数等价与三维旋转（单位四元数）
oMgoal = pin.SE3(pin.Quaternion(-0.5, 0.58, -0.39, 0.52).normalized().matrix(), np.array([1.2, .4, .7]))
viz.addBox('goal', [.1,.1,.1], [ .1,.1,.5, .6])
viz.applyConfiguration('goal', oMgoal)

q0 = np.array([ 0.  ,  0.  ,  1.  ,  0.  ,  0.18,  1.37, -0.24, -0.98,  0.98,
                0.  ,  0.  ,  0.  ,  0.  , -0.13,  0.  ,  0.  ,  0.  ,  0.  ])
DT = 1e-2

# opt process
q = q0.copy()
herr = []

for i in range(500):
    pin.framesForwardKinematics(robot.model, robot.data, q)
    pin.computeJointJacobians(robot.model, robot.data, q)

    oMtool = robot.data.oMf[IDX_TOOL]

    o_Jtool3 = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_TOOL, pin.LOCAL_WORLD_ALIGNED)[:3, :]

    o_TG = oMtool.translation - oMgoal.translation
    # { pinv(J) @ e } 是 Jx - e = 0 的解，但是o_TG以 goal 为基准了，因此下面的 vq 取了一个负号
    vq = - pinv(o_Jtool3) @ o_TG

    q = pin.integrate(robot.model, q, vq * DT)

    viz.display(q)

    time.sleep(1e-3)

    herr.append(o_TG)

# 放置末端执行器 6D
toolMgoal = oMtool.inverse() * oMgoal
tool_w = pin.log(toolMgoal).vector
q = q0.copy()
herr = []
for i in range(500):  # Integrate over 2 second of robot life

    # Run the algorithms that outputs values in robot.data
    pin.framesForwardKinematics(robot.model,robot.data,q)
    pin.computeJointJacobians(robot.model,robot.data,q)

    # Placement from world frame o to frame f oMtool
    oMtool = robot.data.oMf[IDX_TOOL]

    # 6D error between the two frame
    tool_nu = pin.log(oMtool.inverse()*oMgoal).vector

    # Get corresponding jacobian
    tool_Jtool = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_TOOL)

    # Control law by least square
    vq = pinv(tool_Jtool)@tool_nu

    q = pin.integrate(robot.model,q, vq * DT)
    viz.display(q)
    time.sleep(1e-3)

    herr.append(tool_nu)

plt.subplot(211)
plt.plot([ e[:3] for e in herr])
plt.xlabel('control cycle (iter)')
plt.ylabel('error (m)')
plt.subplot(212)
plt.plot([ e[3:] for e in herr])
plt.xlabel('control cycle (iter)')
plt.ylabel('error (rad)')

input('Press ENTER to exit.')