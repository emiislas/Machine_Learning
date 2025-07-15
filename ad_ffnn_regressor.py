import numpy as np
import matplotlib.pyplot as plt
import torch

class FFNeuralNetwork:
    '''
    x1 \in R^{nxd}
    w1 \in R^{dxm}
    b1 \in R_0^{m}
    w2 \in R_0^{mxout}
    b2 \ in R^{out}
    '''
    def __init__(self, input_width, learning_rate, hidden_width, output_width=1):
        self.w1 = torch.randn(input_width, hidden_width, requires_grad=True)
        self.b1 = torch.zeros(hidden_width, requires_grad=True) 
        self.w2 = torch.randn(hidden_width, output_width, requires_grad=True)
        self.b2 = torch.zeros(output_width, requires_grad=True)
        self.alpha = learning_rate


    def relu(self, x):
        return torch.maximum(torch.zeros_like(x), x)

    def loss(self,y_true,y_pred):
        return torch.mean((y_pred-y_true)**2)

    def dloss(self,y_true,y_pred):
        return -2*((y_pred-y_true)/len(y_true))

    def forward(self, x):
        z1 = x@self.w1 + self.b1
        a1 = self.relu(z1)
        z2 = a1@self.w2 + self.b2
        a2 = z2

        return a2
    
    def backward(self, y_pred, y_true):
        loss = self.loss(y_true, y_pred)
        loss.backward()
        #we don't want to store the operation
        #so we do no_grad()
        with torch.no_grad():
            self.w1 -= self.alpha * self.w1.grad
            self.b1 -= self.alpha * self.b1.grad
            self.w2 -= self.alpha * self.w2.grad
            self.b2 -= self.alpha * self.b2.grad

            self.w1.grad.zero_()
            self.b1.grad.zero_()
            self.w2.grad.zero_()
            self.b2.grad.zero_()
        return loss

    def train(self, x, y, iters):
        
        plt.plot(y,c = 'k', label=f'true')
        for i in range(iters):
            y_pred = self.forward(x)
            loss_i = self.backward(y_pred, y)
            if i % 500 == 0 and i !=0:
                print(f"[{i}] acc: ", loss_i.detach().numpy())
                plt.plot(y_pred.detach().numpy(),linestyle='--',label=f'pred{i}')


#def func(x1):
    #return np.sin(2*x1) + np.sin(x1) 
#    return np.sin(2*x1)

def func(x1):
    return x**2
#x = np.linspace(0,10,100).reshape(-1,1)
x = torch.linspace(-5,5,100).reshape(-1,1)
y = func(x) 
model = FFNeuralNetwork(input_width = x.shape[1], hidden_width = 50,learning_rate = 1e-5)
model.train(x,y,10000)
plt.legend()
plt.show()

