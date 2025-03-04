import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return (np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1))

def dfdw(x):
    return -(2*x)/((x**2+1)**2)-1/5*(np.exp(-1/10*(x-6)**2)) * (x-6)-2*(np.exp(-(x-2)**2))*(x-2)

xline = np.linspace(-2,10,1000)
xmax = xline[np.argmax(f(xline))]
plt.plot(xline,f(xline))

alpha = 0.5
stochastic_jump = 10

x = np.random.randint(-2,10)

mrk = 'x' 
color = 'red'
for i in range(1000):

    x_new = x + alpha*dfdw(x)
    if ((x_new-x)**2) < 0.1e-6:
        mrk = 'x' 
        color = 'blue'
        x_new = np.random.randint(-2,10)

    plt.scatter(x_new, f(x_new), color=color, marker = mrk)
    loss = ((x-xmax)**2)/2
    if loss < 0.1e-6:
        print("Loss:", loss, "Iterations:", i)
        break


    mrk = 'o'
    color = 'orange'
    x = x_new

plt.show()
