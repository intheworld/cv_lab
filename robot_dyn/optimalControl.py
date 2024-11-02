import crocoddyl
import pinocchio as pin
import numpy as np
import example_robot_data as robex
from utils.meshcat_viewer_wrapper import MeshcatVisualizer
from utils.croco_utils import displayTrajectory


robot = robex.load('talos_arm')
robot_model = robot.model

robot_model.armature =np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.])*5  # It is a regularization of the mass matric M' = M + lamb I
robot_model.q0 = np.array([3.5,2,2,0,0,0,0])
robot_model.x0 = np.concatenate([robot_model.q0, np.zeros(robot_model.nv)])
robot_model.gravity *= 0

viz = MeshcatVisualizer(robot)
viz.display(robot_model.q0)
viz.viewer.open()

viz.addBox('world/goal',[.1,.1,.1],[0,1,0,1])
viz.applyConfiguration('world/goal',[.2,.5,.5,0,0,0,1])

FRAME_TIP = robot_model.getFrameId("gripper_left_fingertip_3_link")  # Wrapper around pinocchio
goal = np.array([.2,.5,.5])

state = crocoddyl.StateMultibody(robot_model)

runningCostModel = crocoddyl.CostModelSum(state)  # The state will remain the same during running
terminalCostModel = crocoddyl.CostModelSum(state)  # And also for final state

# Cost for 3d tracking || p(q) - pref ||**2
goalTrackingRes = crocoddyl.ResidualModelFrameTranslation(state,FRAME_TIP, goal)
goalTrackingCost = crocoddyl.CostModelResidual(state,goalTrackingRes)

# Cost for 6d tracking  || log( M(q)^-1 Mref ) ||**2
Mref = pin.SE3(pin.utils.rpyToMatrix(0, np.pi/2, -np.pi/2), goal)
goal6TrackingRes = crocoddyl.ResidualModelFramePlacement(state,FRAME_TIP, Mref)
goal6TrackingCost = crocoddyl.CostModelResidual(state,goal6TrackingRes)

# Cost for state regularization || x - x* ||**2
xRegWeights = crocoddyl.ActivationModelWeightedQuad(np.array([1,1,1,1,1,1,1, 1,1,1,1,2,2,2.]))
xRegRes = crocoddyl.ResidualModelState(state,robot_model.x0)
xRegCost = crocoddyl.CostModelResidual(state,xRegWeights,xRegRes)

# Cost for control regularization || u - g(q) ||**2
uRegRes = crocoddyl.ResidualModelControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state,uRegRes)

# Terminal cost for state regularization || x - x* ||**2
xRegWeightsT=crocoddyl.ActivationModelWeightedQuad(np.array([1,1,1,1,1,1,1, 1,1,1,1,2,2,2.]))
xRegResT = crocoddyl.ResidualModelState(state,robot_model.x0)
xRegCostT = crocoddyl.CostModelResidual(state,xRegWeightsT,xRegResT)

runningCostModel.addCost("gripperPose", goalTrackingCost, .001)  # addCost is a method of CostModelSum that take another CostModel
runningCostModel.addCost("xReg", xRegCost, 1e-3)  # We also weight the sum of cost
runningCostModel.addCost("uReg", uRegCost, 1e-6)
# 计算区间应该不一样
terminalCostModel.addCost("gripperPose", goalTrackingCost, 10)
terminalCostModel.addCost("xReg", xRegCostT, .01)

actuationModel = crocoddyl.ActuationModelFull(state)
dt = 1e-2
# A step in the running step
## We use the freefall forward dynamic as dotx = f(x,u) it could be a contact one (as in prev TP) and so on
### It is the differential model chosen around the state
## We precise the duration of the step dt
## We precise the integration scheme we use
### It is the integrated model chosen around the differential model
## We precise which cost is used during this step
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, runningCostModel), dt)
runningModel.differential.armature = robot_model.armature


terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, terminalCostModel), 0.)
terminalModel.differential.armature = robot_model.armature

T = 100
problem = crocoddyl.ShootingProblem(robot_model.x0, [runningModel] * T, terminalModel)

ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
])

ddp.solve([],[],1000)  # xs_init,us_init,maxiter

displayTrajectory(viz, ddp.xs, ddp.problem.runningModels[0].dt, 12)

log = ddp.getCallbacks()[0]
crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1)

print(f'term final state = {ddp.xs[-1]}')


