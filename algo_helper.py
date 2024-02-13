import numpy as np
from algorithm.sgd import SGD
from data_helper import get_data
from typing import Type
from metric_function import MetricFunction

def run_sgd(data_file: str, loss_cls: Type[MetricFunction], metric_cls: Type[MetricFunction], lr: float, batch_size: int):
    X_train, Y_train, X_test, Y_test = get_data(data_file)
    loss_fn = loss_cls(Y=Y_train)
    # take values from gaussian distribution
    Theta = np.random.normal(0, 1, loss_fn.get_theta_shape(X_train))
    metric_fn = metric_cls(Theta, Y_train)
    # X: (batch_size, n), Y: (batch_size, m), Theta: (n, m)
    # print(f'{X_train.shape=}, {Y_train.shape=}, {Theta.shape=}')
    # train
    my_sgd = SGD(loss_fn, metric_fn, lr=lr, stop_condition=0.0005, batch_size=batch_size, log=False)
    Theta, accuracy_train, accuracy_test = my_sgd.run((X_train, Y_train), Theta, (X_test, Y_test))
    return Theta, accuracy_train, accuracy_test
