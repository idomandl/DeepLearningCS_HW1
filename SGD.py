import math
import numpy as np
import matplotlib.pyplot as plt

class SGD:
    def __init__(self, loss_fn, lr=0.01, stop_condition=0.0001, batch_size=100, loss_sample_size=50):
        self.loss_fn = loss_fn
        self.learning_rate = lr
        self.stop_condition = stop_condition
        self.batch_size = batch_size
        self.loss_sample_size = loss_sample_size

    def run(self, D_train, Theta, D_test):
        X_train, Y_train = D_train
        X_test, Y_test = D_test
        losses_train, losses_test = [], []
        grad_tot = np.inf
        prev_grad_tot = 0
        # each iteration is an epoch
        while abs(grad_tot - prev_grad_tot) > self.stop_condition:
            prev_grad_tot = grad_tot
            loss_tot = 0
            grad_tot = 0
            # each iteration is a batch
            for i in range(0, X_train.shape[0], self.batch_size):
                X_batch = X_train[i:i+self.batch_size]
                Y_batch = Y_train[i:i+self.batch_size]
                # calculate loss for train set
                indices = np.random.choice(X_batch.shape[0], self.loss_sample_size, replace=False)
                X_train_sample = X_batch[indices]
                Y_train_sample = Y_batch[indices]
                loss_tot += self.loss_fn(X_train_sample, Y_train_sample, Theta)
                # calculate gradient
                grad = self.loss_fn.calc_grad(X_batch, Y_batch, Theta)
                Theta_change = -self.learning_rate * grad
                grad_tot += np.linalg.norm(Theta_change)
                Theta += Theta_change
            losses_train.append(loss_tot / math.ceil(X_train.shape[0] / self.batch_size))
            # calculate loss for test set
            indices = np.random.choice(X_test.shape[0], self.loss_sample_size, replace=False)
            X_test_sample = X_test[indices]
            Y_test_sample = Y_test[indices]
            losses_test.append(self.loss_fn(X_test_sample, Y_test_sample, Theta))
            if len(losses_train) % 200 == 0:
                print(f'Loss train: {losses_train[-1]} , Loss test: {losses_test[-1]}')
        return Theta, losses_train, losses_test

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
        # 0.5 * |Wx-b|^2
        return 0.5 * np.linalg.norm(X @ Theta - Y) ** 2 / X.shape[0] #? is it ok to divide by X.shape[0]?

    def calc_grad(self, X, Y, Theta):
        # return sum(x.T @ (x @ Theta - y) for x, y in zip(X, Y)) / X.shape[0]
        # (n, batch_size) @ (batch_size, m) = (n, m)
        return X.T @ (X @ Theta - Y) / X.shape[0] #? is it ok to divide by X.shape[0]?
    
    def get_theta_shape(self, X, Y):
        return (X.shape[1], Y.shape[1])

class SoftmaxLoss(LossFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, X, Y, Theta):
        max_element = (X @ Theta).max(axis=1, keepdims=True)
        # sum((batch_size, m) * (batch_size, m)) / batch_size
        return np.sum(Y * np.log(np.exp(X @ Theta - max_element) / sum(np.exp(X @ Theta_i.T[:,None] - max_element) for Theta_i in Theta.T))) / X.shape[0]

    def calc_grad(self, X, Y, Theta):
        max_element = (X @ Theta).max(axis=1, keepdims=True)
        # (n, batch_size) @ (batch_size, m) = (n, m)
        return X.T @ (np.exp(X @ Theta - max_element) / sum(np.exp(X @ Theta_i.T[:,None] - max_element) for Theta_i in Theta.T) - Y) / X.shape[0]
    
    def get_theta_shape(self, X, Y):
        return (X.shape[1], Y.shape[1])


import scipy.io as sio

DATA_FILES = ['GMMData.mat', 'PeaksData', 'SwissRollData.mat']

def shuffle_data(X, Y):
    indices = np.random.permutation(X.shape[0])
    return X[indices], Y[indices]

def add_bias(X):
    return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

fig, axs = plt.subplots(ncols=len(DATA_FILES))
fig.suptitle('Losses')
fig.tight_layout(pad=3.0)

for i, data_file in enumerate(DATA_FILES):
    print(f'Processing {data_file}')
    # open .mat file
    mat_contents = sio.loadmat('SwissRollData.mat')
    X_train = mat_contents['Ct'].T
    Y_train = mat_contents['Yt'].T
    X_test = mat_contents['Cv'].T
    Y_test = mat_contents['Yv'].T

    # Shuffle the data
    X_train, Y_train = shuffle_data(X_train, Y_train)
    X_test, Y_test = shuffle_data(X_test, Y_test)

    # Add bias term
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)

    # X: (batch_size, n), Y: (batch_size, m), Theta: (n, m)
    loss_fn = LinearLeastSquaresLoss()
    # take values from gaussian distribution
    Theta = np.random.normal(0, 1, loss_fn.get_theta_shape(X_train, Y_train))
    print(f'{X_train.shape=}, {Y_train.shape=}, {Theta.shape=}')

    # train
    my_sgd = SGD(loss_fn, lr=0.001, stop_condition=0.0000001, batch_size=1000)
    Theta, loss_train, loss_test = my_sgd.run((X_train, Y_train), Theta, (X_test, Y_test))
    print(Theta)
    print(f'{Theta.shape=}')
    print(f'last loss_train: {loss_train[-1]}, last loss_test: {loss_test[-1]}')
    axs[i].plot(loss_train, label='train')
    axs[i].plot(loss_test, label='test', alpha=0.6)
    axs[i].legend()
    axs[i].set_title(data_file)

plt.show()
