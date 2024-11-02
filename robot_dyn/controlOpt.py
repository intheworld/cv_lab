import numpy as np

# 最优控制的简单例子
x = np.random.rand(3) # x, y, t(角度)
u = np.random.rand(2) # v(速度), w（转角）

v, w = u
c, s = np.cos(x[2]), np.sin(x[2])
dt = 1e-2
dx = np.array([ v * c, v * s, w ])
xnext = x + dx * dt


# 损失函数
stateWeight = 1
ctrlWeight = 1
costResidual = np.concatenate([ np.multiply(x, stateWeight), np.multiply(u, ctrlWeight)], axis = None)
cost = .5 * sum(costResidual**2)
print(f'cost = {cost}')

import crocoddyl
import matplotlib.pylab as plt
from utils.unicycle_utils import plotUnicycle


model = crocoddyl.ActionModelUnicycle()
data = model.createData()

model.costWeights = np.array([1, 1])

x0 = np.matrix([ -1., -1., 1. ]).T #x,y,theta
T  = 20
problem = crocoddyl.ShootingProblem(x0, [ model ] * T, model)

us = [ np.matrix([1., 1.]).T for t in range(T) ]
xs = problem.rollout(us)

for x in xs:
    plotUnicycle(x)
plt.axis([-2,2.,-2.,2.])
plt.show()


ddp = crocoddyl.SolverDDP(problem)
done = ddp.solve()
assert done

plt.clf()
for x in ddp.xs: plotUnicycle(x)
plt.axis([-2,2,-2,2])
plt.show()
print(f'final state = {ddp.xs[-1]}')


model_term = crocoddyl.ActionModelUnicycle()
model_term.costWeights = np.matrix([
    100,   # state weight
    0  # control weight
]).T

# Define integral+terminal models
problem = crocoddyl.ShootingProblem(x0, [ model ] * T, model_term)
ddp = crocoddyl.SolverDDP(problem)

ddp.solve()
for x in ddp.xs:
    plotUnicycle(x)
plt.axis([-2,2.,-2.,2.])
plt.show()

print(f'term final state = {ddp.xs[-1]}')
