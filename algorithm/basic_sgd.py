import numpy as np
import math
from function.function import Function


class SGD:
    def __init__(self, loss: Function, metric_fn, lr=0.01, stop_condition=0.0005, batch_size=100, metric_sample_percentage=0.3, log=True):
        self.loss = loss
        self.metric_fn = metric_fn
        self.learning_rate = lr
        self.stop_condition = stop_condition
        self.batch_size = batch_size
        self.metric_sample_percentage = metric_sample_percentage
        self.log = log
        self.compare_window = 50

    def run(self, D_train, Theta, D_test):
        X_train, Y_train = D_train
        X_test, Y_test = D_test
        metric_train, metric_test, losses = [], [], []
        # each iteration is an epoch
        while len(losses) < self.compare_window or max(abs(losses[-self.compare_window:] - losses[-1])) > self.stop_condition:
            # train
            Theta, metric_avg, loss_avg = self.handle_batches(X_train, Y_train, Theta)
            losses.append(loss_avg)
            # calculate metric for train set
            metric_train.append(metric_avg)
            # calculate metric for test set
            X_test_sample, Y_test_sample = self.select_metric_sample(X_test, Y_test)
            metric_test.append(self.metric_fn(X_test_sample, Y_test_sample, Theta))
            if self.log and len(metric_train) % 200 == 0:
                print(f'train = {metric_train[-1]} , test = {metric_test[-1]}')
        return Theta, metric_train, metric_test
    
    def handle_batches(self, X_train, Y_train, Theta):
        metric_tot = grad_tot = loss_tot = 0
        # each iteration is a batch
        for i in range(0, X_train.shape[0], self.batch_size):
            X_batch = X_train[i:i + self.batch_size]
            Y_batch = Y_train[i:i + self.batch_size]
            # calculate metric for train set
            Theta, loss, grad, metric = self.handle_batch(X_batch, Y_batch, Theta)
            loss_tot += loss
            metric_tot += metric
            # grad_tot += grad
        metric_avg = metric_tot / math.ceil(X_train.shape[0] / self.batch_size)
        loss_avg = loss_tot / math.ceil(X_train.shape[0] / self.batch_size)
        # grad_avg = grad_tot / math.ceil(X_train.shape[0] / self.batch_size)
        return Theta, metric_avg, loss_avg

    def handle_batch(self, X_batch, Y_batch, Theta):
        # calculate loss for train set
        X_train_sample, Y_train_sample = self.select_metric_sample(X_batch, Y_batch)
        metric = self.metric_fn(X_train_sample, Y_train_sample, Theta)
        # calculate gradient
        loss = self.loss.loss(X_batch, Y_batch, Theta)
        grad = self.loss.loss_grad_Theta(X_batch, Y_batch, Theta)
        Theta_change = -self.learning_rate * grad
        new_Theta = Theta + Theta_change
        return new_Theta, loss, np.linalg.norm(Theta_change), metric
    
    def select_metric_sample(self, X, Y):
        indices = np.random.choice(X.shape[0], int(X.shape[0] * self.metric_sample_percentage), replace=False)
        return X[indices], Y[indices]
