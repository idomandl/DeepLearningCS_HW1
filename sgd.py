import math
import numpy as np
import matplotlib.pyplot as plt
from softmax_loss import SoftmaxLoss, SoftmaxLossLoop
from linear_least_squares_loss import LinearLeastSquaresLoss

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
                self.loss_fn.set_Theta(Theta)
                self.loss_fn.set_Y(Y_train_sample)
                loss_tot += self.loss_fn(X_train_sample)
                # calculate gradient
                self.loss_fn.set_Theta(Theta)
                self.loss_fn.set_Y(Y_batch)
                grad = self.loss_fn.calc_grad(X_batch)
                Theta_change = -self.learning_rate * grad
                grad_tot += np.linalg.norm(Theta_change)
                Theta += Theta_change
            losses_train.append(loss_tot / math.ceil(X_train.shape[0] / self.batch_size))
            # calculate loss for test set
            indices = np.random.choice(X_test.shape[0], self.loss_sample_size, replace=False)
            X_test_sample = X_test[indices]
            Y_test_sample = Y_test[indices]
            self.loss_fn.set_Theta(Theta)
            self.loss_fn.set_Y(Y_test_sample)
            losses_test.append(self.loss_fn(X_test_sample))
            if len(losses_train) % 200 == 0:
                print(f'Loss train: {losses_train[-1]} , Loss test: {losses_test[-1]}')
        return Theta, losses_train, losses_test

import scipy.io as sio

DATA_FILES = ['GMMData.mat', 'PeaksData', 'SwissRollData.mat']

def shuffle_data(X, Y):
    indices = np.random.permutation(X.shape[0])
    return X[indices], Y[indices]

def add_bias(X):
    return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

def main():

    for loss in [SoftmaxLoss, LinearLeastSquaresLoss, SoftmaxLossLoop]:
        fig, axs = plt.subplots(ncols=len(DATA_FILES))
        fig.suptitle(f'Losses, {loss.name}')
        fig.tight_layout(pad=3.0)
        for i, data_file in enumerate(DATA_FILES):
            print(f'Processing {data_file}')
            # open .mat file
            mat_contents = sio.loadmat(f'{data_file}')
            Y_train = mat_contents['Ct'].T
            X_train = mat_contents['Yt'].T
            Y_test = mat_contents['Cv'].T
            X_test = mat_contents['Yv'].T
            # Shuffle the data
            X_train, Y_train = shuffle_data(X_train, Y_train)
            X_test, Y_test = shuffle_data(X_test, Y_test)

            # Add bias term
            X_train = add_bias(X_train)
            X_test = add_bias(X_test)

            # X: (batch_size, n), Y: (batch_size, m), Theta: (n, m)
            loss_fn = loss(0,np.random.randn(Y_train.shape[0],Y_train.shape[1]))
            # take values from gaussian distribution
            Theta = np.random.normal(0, 1, loss_fn.get_theta_shape(X_train))
            #print(f'{X_train.shape=}, {Y_train.shape=}, {Theta.shape=}')

            # train
            my_sgd = SGD(loss_fn, lr=0.001, stop_condition=0.000001, batch_size=1000)
            Theta, loss_train, loss_test = my_sgd.run((X_train, Y_train), Theta, (X_test, Y_test))
            print(Theta)
            #print(f'{Theta.shape=}')
            print(f'last loss_train: {loss_train[-1]}, last loss_test: {loss_test[-1]}')
            axs[i].plot(loss_train, label='train')
            axs[i].plot(loss_test, label='test', alpha=0.6)
            axs[i].legend()
            axs[i].set_title(data_file)
        plt.show()
if __name__ == '__main__':
    main()