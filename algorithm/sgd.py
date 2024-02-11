import numpy as np
import math


class SGD:
    def __init__(self, loss_fn, metric_fn, lr=0.01, stop_condition=0.0001, batch_size=100, metric_sample_size=50):
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.learning_rate = lr
        self.stop_condition = stop_condition
        self.batch_size = batch_size
        self.metric_sample_size = metric_sample_size

    def run(self, D_train, Theta, D_test):
        X_train, Y_train = D_train
        X_test, Y_test = D_test
        metric_train, metric_test = [], []
        grad_tot = np.inf
        prev_grad_tot = 0
        # each iteration is an epoch
        while abs(grad_tot - prev_grad_tot) > self.stop_condition:
            prev_grad_tot = grad_tot
            # train
            Theta, metric_tot, grad_tot = self.handle_batches(X_train, Y_train, Theta)
            # calculate metric for train set
            metric_train.append(metric_tot / math.ceil(X_train.shape[0] / self.batch_size))
            # calculate metric for test set
            X_test_sample, Y_test_sample = self.select_metric_sample(X_test, Y_test)
            metric_test.append(self.metric_fn(X_test_sample, Y=Y_test_sample, Theta=Theta))
            if len(metric_train) % 200 == 0:
                print(f'{self.metric_fn.name}: train = {metric_train[-1]} , test = {metric_test[-1]}')
        return Theta, metric_train, metric_test
    
    def handle_batches(self, X_train, Y_train, Theta):
        metric_tot = grad_tot = 0
        # each iteration is a batch
        for i in range(0, X_train.shape[0], self.batch_size):
            X_batch = X_train[i:i + self.batch_size]
            Y_batch = Y_train[i:i + self.batch_size]
            # calculate metric for train set
            Theta, metric, grad = self.handle_batch(X_batch, Y_batch, Theta)
            metric_tot += metric
            grad_tot += grad
        return Theta, metric_tot, grad_tot

    def handle_batch(self, X_batch, Y_batch, Theta):
        # calculate loss for train set
        X_train_sample, Y_train_sample = self.select_metric_sample(X_batch, Y_batch)
        metric = self.metric_fn(X_train_sample, Y=Y_train_sample, Theta=Theta)
        # calculate gradient
        grad = self.loss_fn.calc_grad(X_batch, Y=Y_batch, Theta=Theta)
        Theta_change = -self.learning_rate * grad
        return Theta + Theta_change, metric, np.linalg.norm(Theta_change)
    
    def select_metric_sample(self, X, Y):
        indices = np.random.choice(X.shape[0], self.metric_sample_size, replace=False)
        return X[indices], Y[indices]
