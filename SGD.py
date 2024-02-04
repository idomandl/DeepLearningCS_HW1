import numpy as np
import matplotlib.pyplot as plt

class SGD:
    def __init__(self, loss_fn, lr=0.01, stop_condition=0.0001, batch_size=100):
        self.loss_fn = loss_fn
        self.learning_rate = lr
        self.stop_condition = stop_condition
        self.batch_size = batch_size

    def run(self, X, Y, Theta):
        losses = []
        grad_tot = np.inf
        # each iteration is an epoch
        while len(losses) <= 5 or abs(losses[-1] - losses[-5]) > self.stop_condition:
            loss_tot = 0
            grad_tot = 0
            # each iteration is a batch
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i+self.batch_size]
                Y_batch = Y[i:i+self.batch_size]
                loss_tot += self.loss_fn(X_batch, Y_batch, Theta)
                grad = self.loss_fn.calc_grad(X_batch, Y_batch, Theta)
                Theta_change = -self.learning_rate * grad
                grad_tot += np.linalg.norm(Theta_change)
                Theta += Theta_change
            losses.append(loss_tot / X.shape[0])
            if len(losses) % 200 == 0:
                print(f'Loss: {losses[-1]}')
        return Theta, losses

class LossFunction:
    def __init__(self):
        pass

    def __call__(self, X, Y, Theta):
        pass

    def calc_grad(self, X, Y, Theta):
        pass

    def get_theta_shape(self, X, Y):
        pass

class LinearLeastSquaresLoss(LossFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, X, Y, Theta):
        # return 0.5 * sum(np.linalg.norm(x @ Theta - y)**2 for x, y in zip(X, Y)) / X.shape[0]
        return 0.5 * np.linalg.norm(X @ Theta - Y) ** 2 / X.shape[0] #? is it ok to divide by X.shape[0]?

    def calc_grad(self, X, Y, Theta):
        # return sum(x.T @ (x @ Theta - y) for x, y in zip(X, Y)) / X.shape[0]
        return X.T @ (X @ Theta - Y) / X.shape[0] #? is it ok to divide by X.shape[0]?
    
    def get_theta_shape(self, X, Y):
        return (X.shape[1], Y.shape[1])


import scipy.io as sio

# open .mat file
mat_contents = sio.loadmat('GMMData.mat')
print(mat_contents.keys())
X = mat_contents['Ct'].T
Y = mat_contents['Yt'].T

# Shuffle the data
indices = np.random.permutation(X.shape[0])
X = X[indices]
Y = Y[indices]

# X: (batch_size, n), Y: (batch_size, m), Theta: (n, m)
loss_fn = LinearLeastSquaresLoss()
Theta = np.zeros(loss_fn.get_theta_shape(X, Y))
print(f'{X.shape=}, {Y.shape=}, {Theta.shape=}')

# train
my_sgd = SGD(loss_fn, lr=0.0001, stop_condition=0.000000001, batch_size=1000)
Theta, loss_grads = my_sgd.run(X, Y, Theta)
print(Theta)
print(f'{Theta.shape=}')
print(f'last loss: {loss_grads[-1]}')
plt.plot(loss_grads)
plt.show()
