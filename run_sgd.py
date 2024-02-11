import numpy as np
import matplotlib.pyplot as plt
from loss.softmax_loss import SoftmaxLoss
from loss.linear_least_squares_loss import LinearLeastSquaresLoss
from accuracy.softmax_accuracy import SoftmaxAccuracy
from algorithm.sgd import SGD
import scipy.io as sio


DATA_FILES = ['GMMData', 'PeaksData', 'SwissRollData']


def shuffle_data(X, Y):
    indices = np.random.permutation(X.shape[0])
    return X[indices], Y[indices]


def add_bias(X):
    return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)


LEARNING_RATES = [0.01, 0.001, 0.0001]
BATCH_SIZES = [10, 50, 100, 500, 1000]
best_accuracy = 0
best_result = 0
for loss in [SoftmaxLoss, LinearLeastSquaresLoss]:
    fig, axs = plt.subplots(ncols=len(DATA_FILES))
    fig.suptitle(f'Accuracy, {loss.name}')
    fig.tight_layout(pad=3.0)
    for i, data_file in enumerate(DATA_FILES):
        for lr in LEARNING_RATES:
            for batch_size in BATCH_SIZES:
                print(f'Processing {data_file}')
                # open .mat file
                mat_contents = sio.loadmat(f'data/{data_file}.mat')
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
                loss_fn = loss(0, np.random.randn(Y_train.shape[0], Y_train.shape[1]))
                # take values from gaussian distribution
                Theta = np.random.normal(0, 1, loss_fn.get_theta_shape(X_train))
                # print(f'{X_train.shape=}, {Y_train.shape=}, {Theta.shape=}')

                # train
                softmax_accuracy = SoftmaxAccuracy(Theta, Y_train)
                my_sgd = SGD(loss_fn, softmax_accuracy, lr=0.001, stop_condition=0.000001, batch_size=1000)
                Theta, accuracy_train, accuracy_test = my_sgd.run((X_train, Y_train), Theta, (X_test, Y_test))
                curr_acc = np.mean(accuracy_test[-10:])
                if curr_acc > best_accuracy:
                    best_accuracy = curr_acc
                    best_result = (Theta, accuracy_train, accuracy_test, loss, lr, batch_size, data_file)
                print(f'last acc_train: {accuracy_train[-1]}, last acc_test: {accuracy_test[-1]}')
        axs[i].plot(best_result[1], label='train')
        axs[i].plot(best_result[2], label='test', alpha=0.6)
        axs[i].legend()
        axs[i].set_title(f"{data_file}, lr={best_result[4]}, batch_size={best_result[5]}")
    plt.show()
