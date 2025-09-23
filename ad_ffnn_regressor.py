import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import Parameter, ParameterList

class FFNeuralNetwork:
    '''
    x1 \in R^{nxd}
    w1 \in R^{dxm}
    b1 \in R_0^{m}
    w2 \in R_0^{mxout}
    b2 \ in R^{out}
    '''
    def __init__(self, input_width, learning_rate, hidden_width, num_layers=1,output_width=1, activation = "relu"):
        self.num_layers = num_layers
        self.w_inp = Parameter(torch.randn(input_width, hidden_width, requires_grad=True))
        self.b_inp = Parameter(torch.zeros(hidden_width, requires_grad=True))
        self.W = ParameterList()
        self.B = ParameterList()
        if num_layers > 1:
            for _ in range(num_layers - 1):
                self.W.append(Parameter(torch.randn(hidden_width, hidden_width, requires_grad=True)))
                self.B.append(Parameter(torch.zeros(hidden_width, requires_grad=True)))

        self.w_out = Parameter(torch.randn(hidden_width, output_width, requires_grad=True))
        self.b_out = Parameter(torch.zeros(output_width, requires_grad=True))
        self.alpha = learning_rate
        self.activation = activation


    def relu(self, x):
        return torch.maximum(torch.zeros_like(x), x)

    def tanh(self, x):
        return torch.tanh(x)

    def compute_loss(self,y_true,y_pred):
        return torch.mean((y_pred-y_true)**2)

    def get_activation(self):
        activations = {
                    "relu" : self.relu,
                    "tanh" : self.tanh
                }

        return activations[self.activation]

    def forward_pass(self, x):
        act_f = self.get_activation()
        z_inp = x@self.w_inp + self.b_inp
        a_inp = act_f(z_inp)

        A = [a_inp]
        Z = []
        if self.num_layers > 1:
            for i in range(self.num_layers-1):
                Z.append(A[i]@self.W[i] + self.B[i])
                A.append(act_f(Z[i]))

        z_out = A[-1]@self.w_out + self.b_out
        a_out = z_out

        return a_out

    def zero_grad(self, tensor):
        if tensor.grad is not None:
            tensor.grad.zero_()

    def backward_pass(self, y_pred, y_true):
        loss = self.compute_loss(y_true, y_pred)
        loss.backward()
        #we don't want to store the operation
        #so we do no_grad()
        with torch.no_grad():
            self.w_inp -= self.alpha * self.w_inp.grad
            self.b_inp -= self.alpha * self.b_inp.grad
            if self.num_layers > 1:
                for i in range(self.num_layers-1):
                    self.W[i] -= self.alpha * self.W[i].grad
                    self.B[i] -= self.alpha * self.B[i].grad
                    self.zero_grad(self.W[i])
                    self.zero_grad(self.B[i])

            self.w_out -= self.alpha * self.w_out.grad
            self.b_out -= self.alpha * self.b_out.grad

            self.zero_grad(self.w_inp)
            self.zero_grad(self.b_inp)
            self.zero_grad(self.w_out)
            self.zero_grad(self.b_out)
        return loss

    def train(self, x, y, iters):
        
        plt.plot(x.detach().numpy(), y.detach().numpy(), c='k', label='true')
        for i in range(iters):
            y_pred = self.forward_pass(x)
            loss_i = self.backward_pass(y_pred, y)
            if i % 500 == 0 and i !=0:
                print(f"[{i}] loss: ", loss_i.detach().numpy())
                plt.plot(x.detach().numpy(), y_pred.detach().numpy(), linestyle='--', label=f'pred{i}')


def func(x1):
    return np.sin(2*x1) + np.sin(x1) 
    #return np.sin(2*x1)

#def func(x):
#    return x**2

x = torch.linspace(0,10,100).reshape(-1,1)
#x = torch.linspace(-5,5,100).reshape(-1,1)
y = func(x) 
model = FFNeuralNetwork(input_width = x.shape[1], hidden_width = 20, num_layers = 3, learning_rate = 1e-3, activation="tanh")
model.train(x,y,20000)
plt.legend()
plt.show()

