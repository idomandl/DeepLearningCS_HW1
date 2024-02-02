import numpy as np
import matplotlib.pyplot as plt

class SGD:

    def __init__(self, calc_grad, lr=0.01, stop_condition=0.0001):
        self.lr = lr
        self.stop_condition = stop_condition
        self.calc_grad = calc_grad

    def run(self, X, Y, Theta):
        loss_gradients = []
        g = self.calc_grad(X, Y, Theta)
        while np.linalg.norm(g) > self.stop_condition:
            loss_gradients.append(np.linalg.norm(g))
            Theta = Theta - self.lr * g
            # get batch from x and y
            g = self.calc_grad(X, Y, Theta)
        return Theta, loss_gradients




class CalcGrad:
    def __init__(self):
        pass

    def __call__(self, X, Y, Theta):
        pass

class LeastSquaresCG(CalcGrad):
    def __init__(self):
        super().__init__()

    def __call__(self, X, Y, Theta):
        return X.T @ (X @ Theta - Y)




test_sgd = SGD(LeastSquaresCG(), lr=0.0001, stop_condition=0.0001)
# open .mat file
import scipy.io as sio
mat_contents = sio.loadmat('GMMData.mat')
print(mat_contents.keys())
X = mat_contents['Ct'][:,:100]
Y = mat_contents['Yt'][:,:100]
print(X.shape, Y.shape)
Theta = np.zeros((X.shape[1], 1))
theta, loss_grads = test_sgd.run(X, Y, Theta)
print(theta)
plt.plot(loss_grads)
plt.show()