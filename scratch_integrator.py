import numpy as np
import matplotlib.pyplot as plt

def ode_analytical_sol(t):
    w = 1.2
    zeta = 0.3/(2*w)
    wd = w*np.sqrt(1-zeta**2)
    c2 = (zeta*w)/wd
    x = np.exp(-zeta*w*t) * (np.cos(wd*t) + c2*np.sin(wd*t))

    xdot = np.exp(-zeta*w*t) * (
        -zeta*w*(np.cos(wd*t) + c2*np.sin(wd*t))
        - wd*np.sin(wd*t) + c2*wd*np.cos(wd*t)
    )
    return x, xdot

def ode_fun(Y):
    x1, x2 = Y
    w = 1.2
    d2xdt2 =  -w**2*x1-0.3*x2
    return np.array([x2, d2xdt2])

def rk4Integrate(interp_fun, y0, dt=0.1, length=0):
    d = int(np.shape(y0)[0])
    n_steps = int(length/dt)
    t = np.arange(0, length+dt, dt)
    y = np.zeros((n_steps+1, d))
    y[0] = y0

    for i in range(n_steps):
        k1 = dt * interp_fun(y[i])
        k2 = dt * interp_fun(y[i] + k1/2)
        k3 = dt * interp_fun(y[i] + k2/2)
        k4 = dt * interp_fun(y[i] + k3)

        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4)/6

    return t, y

y0 = [1, 0]

t, y_hat = rk4Integrate(ode_fun, y0, length = 10)
x_analytical, xdot_analytical = ode_analytical_sol(t)

plt.plot(t, x_analytical, color = "k", label = r"$x$")

plt.plot(t,xdot_analytical, color = "k", label = r"$\dot{x}$")

plt.plot(t, y_hat[:,0], color = "blue", linestyle = ":", label = r"$\hat{x}$")
plt.plot(t, y_hat[:,1], color = "red", linestyle = ":", label = r"$\hat{\dot{x}}$")
plt.legend()
plt.show()


