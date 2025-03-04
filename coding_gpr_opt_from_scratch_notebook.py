
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern
import matplotlib.animation as animation
from PyGPr import GPR, SquaredExponentialKernel

def f(x):
  return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)

x_test = np.linspace(-2,10,100)

def plot_GPRA(ax, data_x, data_y, model, x, kappa=2,visible=True) -> list:
    ax.clear()
    mean = model.predict(x)
    ax.plot(x, f(x))
    ax.plot(x, mean, linestyle= '--', label = 'GP Mean')

    ax.scatter(data_x,data_y, c = 'red', label = "Sampled Points")

    std = np.sqrt(model._memory['variance'])
    ucb = mean + kappa * std
    print(x[np.argmax(ucb)], np.max(ucb))
    plt.scatter(x[np.argmax(ucb)], np.max(ucb), marker='*', c ='red', label="UCB Max")

    #plt.plot(data_x, data_y)
    for i in range(1, kappa+1):
      y_lower=mean - i * std
      y_upper=mean + i * std
      ax.fill_between(x, y_lower, y_upper, color='b', alpha=.1)

    ax.set_xlim(-2, 10)
    ax.set_ylim(0, 2) 

    return x[np.argmax(ucb)]

global_x = [1,4]
fig, ax = plt.subplots()

def update(frame):
  global global_x
  global_y = f(np.array(global_x))
  model = GPR(global_x, global_y, covariance_function=SquaredExponentialKernel(length=2))
  ducb = plot_GPRA(ax, data_x=global_x, data_y=global_y, x=x_test, model=model)
  if frame < 9:
      global_x.append(ducb)


ani = animation.FuncAnimation(fig=fig, func=update, frames=10, interval=1000)
plt.show()
