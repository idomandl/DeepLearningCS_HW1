import numpy as np
import matplotlib.pyplot as plt

class SGD:
    def __init__(self, calc_grad, lr=0.01, stop_condition=0.0001):
        self.calc_grad = calc_grad
        self.lr = lr # learning rate
        self.stop_condition = stop_condition

    def run(self, X, Y, Theta):
        loss_grads = []
        grad = self.calc_grad(X, Y, Theta)
        while np.linalg.norm(grad) > self.stop_condition:
            loss_grads.append(np.linalg.norm(grad))
            # print(f'{Theta.shape=}, {(self.lr * grad).shape=}')
            Theta -= self.lr * grad
            grad = self.calc_grad(X, Y, Theta)
        return Theta, loss_grads

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


import scipy.io as sio

BATCH_SIZE = 100

# open .mat file
mat_contents = sio.loadmat('GMMData.mat')
print(mat_contents.keys())
X = mat_contents['Ct'].T
Y = mat_contents['Yt'].T

# Shuffle the data
indices = np.random.permutation(X.shape[0])
X = X[indices][:BATCH_SIZE]
Y = Y[indices][:BATCH_SIZE]

# X: (batch_size, n), Y: (batch_size, m), Theta: (n, m)
Theta = np.zeros((X.shape[1], Y.shape[1]))
print(f'{X.shape=}, {Y.shape=}, {Theta.shape=}')

# train
my_sgd = SGD(LeastSquaresCG(), lr=0.0001, stop_condition=0.0001)
theta, loss_grads = my_sgd.run(X, Y, Theta)
print(theta)
print(f'{theta.shape=}')
plt.plot(loss_grads)
plt.show()
