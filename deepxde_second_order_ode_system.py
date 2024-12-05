import deepxde as dde
from sympy import *
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import torch

def torch_loss(y_true, y_pred):
    #return pt.nn.HuberLoss()(y_true, y_pred)
    return pt.nn.SmoothL1Loss()(y_true, y_pred)

'''
y''(1) + 2y(1) + y(2) + 0.3y'(1) = 0
y''(2) + 2y(2) + y(1) + 0.3y'(2) = 0
'''

def ode_system(x, y):
    y1, y2 = y[:,[0]], y[:,[1]]

    dy1_dt = dde.grad.jacobian(y1, x)
    d2y1_dt2 = dde.grad.hessian(y1, x)
    dy2_dt = dde.grad.jacobian(y2, x)
    d2y2_dt2 = dde.grad.hessian(y2, x)
    return [d2y1_dt2 + (2*y1) - y2 + (0.03*dy1_dt) - torch.sin(1.2*x) , d2y2_dt2 + (2*y2) - y1 + (0.03*dy2_dt)] 


def func(x):
    #Solving system with Sympy
    y1, y2 = symbols("y1 y2", cls=Function)
    t = symbols("t")
    eqs = [Eq(y1(t).diff(t,t) + 2*y1(t) - y2(t) + 0.03*y1(t).diff(t) - sin(1.2*t),0), 
           Eq(y2(t).diff(t,t) + 2*y2(t) - y1(t) + 0.03*y2(t).diff(t),0)
           ]
    print("Solving ODE: \n")
    pprint(eqs)
    sol = dsolve(eqs, [y1(t),y2(t)], ics={y1(0):1, y1(t).diff(t).subs(t,0):0, y2(t).diff(t).subs(t,0):0, y2(0):0})
    
    f = []
    for i in range(2):
        f.append(lambdify(t,sol[i].rhs, "numpy")(x))

    f = np.array(f).T[0]

    return f    


geom = dde.geometry.TimeDomain(0, 50)

def boundary_l(x, on_initial):
    return on_initial and dde.utils.isclose(x[0], 0)

def bc_func1(inputs, outputs, x):
    return dde.grad.jacobian(outputs, inputs, i=0, j=0)

def bc_func2(inputs, outputs, x):
    return dde.grad.jacobian(outputs, inputs, i=1, j=0)


ic1 = dde.icbc.IC(geom, lambda x: 1, lambda _, on_initial: on_initial, component=0)
ic2 = dde.icbc.IC(geom, lambda x: 0, lambda _, on_initial: on_initial, component=1)
dic1 = dde.icbc.OperatorBC(geom, bc_func1, boundary_l)
dic2 = dde.icbc.OperatorBC(geom, bc_func2, boundary_l)

data = dde.data.TimePDE(geom, ode_system, [ic1, ic2, dic1, dic2], num_domain=500, num_boundary=2, solution=func, num_test=500)

net = dde.nn.FNN([1] + [50] * 3 + [2], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile(
    "adam", lr=0.00001,loss=torch_loss, metrics=["l2 relative error"]
)

losshistory, train_state = model.train(iterations=int(4e4))



variables = ['y1', 'y2']
colors = ['red', 'blue']
for idx, var in enumerate(variables):
    plt.plot(train_state.y_test[:,idx], c='k')
    plt.plot(train_state.y_pred_test[:,idx], c=colors[idx], linestyle='--',label = var)

plt.legend()
plt.show()
dde.saveplot(losshistory, train_state,issave=False, isplot=True)



