import numpy as np
import matplotlib.pyplot as plt

class FFNeuralNetwork:
    '''
    x1 \in R^{nxd}
    w1 \in R^{dxm}
    b1 \in R_0^{m}
    w2 \in R_0^{mxout}
    b2 \ in R^{out}
    '''
    def __init__(self, input_width, learning_rate, hidden_width, output_width=1):
        self.w1 = np.random.randn(input_width, hidden_width) - 0.5
        self.b1 = np.zeros(hidden_width) 
        self.w2 = np.random.randn(hidden_width, output_width) - 0.5
        self.b2 = np.zeros(output_width)
        self.alpha = learning_rate

    def tanh(self, x):
        return np.tanh(x)

    def back_tanh(self, x):
        return 1 - np.tanh(x)**2

    def relu(self, x): 
        return np.maximum(0,x)

    def back_relu(self,x):
        return x>0

    def loss(self,y_true,y_pred):
        return np.mean((y_pred-y_true)**2)

    def dloss(self,y_true,y_pred):
        return -2*((y_pred-y_true)/len(y_true))


    def forward(self, x):
        z1 = x@self.w1 + self.b1
        a1 = self.relu(z1)
        z2 = a1@self.w2 + self.b2
        a2 = z2
        cache = {'z1':z1,'a1':a1,'z2':z2,'x':x}

        return a2, cache


    
    def backward(self, a2, y_true, cache):
        dz2 = self.dloss(a2,y_true)
        dw2 = cache['a1'].T@dz2
        db2 = np.sum(dz2, axis=0)
        dz1 = dz2@self.w2.T * self.back_relu(cache['z1'])
        dw1 = cache['x'].T@dz1    
        db1 = np.sum(dz1, axis=0)

        return dw2, db2, dw1, db1

    def update_params(self, dw2, db2, dw1, db1):
        self.w1 -= self.alpha * dw1
        self.b1 -= self.alpha * db1    
        self.w2 -= self.alpha * dw2  
        self.b2 -= self.alpha * db2    

    def train(self, x, y, iters):
        plt.plot(y,c = 'k', label=f'true')
        for i in range(iters):
            a2, cache = self.forward(x)
            dw2, db2, dw1, db1 = self.backward(a2, y, cache)
            self.update_params(dw2, db2, dw1, db1)
            if i % 500 == 0 and i !=0:
                print(f"[{i}] acc: ", self.loss(y, a2))
                plt.plot(a2,linestyle='--',label=f'pred{i}')


#def func(x1):
    #return np.sin(2*x1) + np.sin(x1) 
#    return np.sin(2*x1)

def func(x1):
    return x**2
#x = np.linspace(0,10,100).reshape(-1,1)
x = np.linspace(-5,5,100).reshape(-1,1)
y = func(x) 
model = FFNeuralNetwork(input_width = x.shape[1], hidden_width = 50,learning_rate = 1e-5)
model.train(x,y,10000)
plt.legend()
plt.show()

