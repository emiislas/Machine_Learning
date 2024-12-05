import numpy as np
import matplotlib.pyplot as plt

class FFNeuralNetwork:
    def __init__(self, input_size, learning_rate, hidden_size, output_size=1):
        self.w1 = np.random.randn(input_size, hidden_size) - 0.5
        self.b1 = np.zeros(hidden_size) 
        self.w2 = np.random.randn(hidden_size, output_size) - 0.5
        self.b2 = np.zeros(output_size)
        self.alpha = learning_rate

    def relu(self, x): 
        return np.maximum(0,x)

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

    def back_relu(self,x):
        return x>0


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


def func(x):
    return np.sin(x) 

x = np.random.randn(100, 1) - 0.5
y = func(x) 
plt.plot(y)

model = FFNeuralNetwork(input_size = x.shape[1], hidden_size = 100,learning_rate = 0.0001)
model.train(x,y,10000)
plt.legend()
plt.show()

