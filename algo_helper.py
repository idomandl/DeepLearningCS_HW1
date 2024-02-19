from algorithm.basic_sgd import SGD as Basic_SGD
from algorithm.sgd import SGD
from data_helper import get_data
from function.function import Function
from nn import NN

def run_sgd(data_file: str, loss: Function, metric_fn, lr: float, batch_size: int):
    X_train, Y_train, X_test, Y_test = get_data(data_file)
    # take values from gaussian distribution
    Theta = loss.generate_Theta(*loss.get_Theta_shape(X_train, Y_train))
    # X: (batch_size, n), Y: (batch_size, m), Theta: (n, m)
    # print(f'{X_train.shape=}, {Y_train.shape=}, {Theta.shape=}')
    # train
    my_sgd = Basic_SGD(loss, metric_fn, lr=lr, stop_condition=0.0005, batch_size=batch_size, log=False)
    Theta, accuracy_train, accuracy_test = my_sgd.run((X_train, Y_train), Theta, (X_test, Y_test))
    return Theta, accuracy_train, accuracy_test

def run_sgd_nn(nn: NN, data_file: str, loss: Function, metric_fn, lr: float, batch_size: int):
    X_train, Y_train, X_test, Y_test = get_data(data_file)
    # take values from gaussian distribution
    Theta = loss.generate_Theta(*loss.get_Theta_shape(X_train, Y_train))
    # X: (batch_size, n), Y: (batch_size, m), Theta: (n, m)
    # print(f'{X_train.shape=}, {Y_train.shape=}, {Theta.shape=}')
    # train
    my_sgd = SGD(nn, loss, metric_fn, lr=lr, stop_condition=0.0005, batch_size=batch_size, log=False)
    Theta, accuracy_train, accuracy_test = my_sgd.run((X_train, Y_train), Theta, (X_test, Y_test))
    return Theta, accuracy_train, accuracy_test
