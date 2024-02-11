import numpy as np
import matplotlib.pyplot as plt
from loss.softmax_loss import SoftmaxLoss
from loss.linear_least_squares_loss import LinearLeastSquaresLoss
from accuracy.softmax_accuracy import SoftmaxAccuracy
from algorithm.sgd import SGD
import scipy.io as sio
from p_tqdm import p_map
# from multiprocessing import Pool
# from tqdm.auto import tqdm

DATA_FILES = ['GMMData', 'PeaksData', 'SwissRollData']

def shuffle_data(X, Y):
    indices = np.random.permutation(X.shape[0])
    return X[indices], Y[indices]


def add_bias(X):
    return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

def run_sgd(data_file, loss, lr, batch_size):
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
    loss_fn = loss(Y=Y_train)
    # take values from gaussian distribution
    Theta = np.random.normal(0, 1, loss_fn.get_theta_shape(X_train))
    # print(f'{X_train.shape=}, {Y_train.shape=}, {Theta.shape=}')
    # train    
    softmax_accuracy = SoftmaxAccuracy(Theta, Y_train)
    my_sgd = SGD(loss_fn, softmax_accuracy, lr=lr, stop_condition=0.0005, batch_size=batch_size, log=False)
    Theta, accuracy_train, accuracy_test = my_sgd.run((X_train, Y_train), Theta, (X_test, Y_test))
    return Theta, accuracy_train, accuracy_test

def generate_sgd_per_hyperparams(loss, lr, batch_size):
    acc_tot = 0
    # results = p_map(run_sgd, DATA_FILES, [loss]*len(DATA_FILES), [lr]*len(DATA_FILES), [batch_size]*len(DATA_FILES))
    accuracy_trains = []
    accuracy_tests = []
    for data_file in DATA_FILES:
        # print(f'Processing {data_file}')
        Theta, accuracy_train, accuracy_test = run_sgd(data_file, loss, lr, batch_size)
        acc_tot += np.median(accuracy_test[-10:])
        accuracy_trains.append(accuracy_train)
        accuracy_tests.append(accuracy_test)
    return acc_tot, accuracy_trains, accuracy_tests


if __name__ == "__main__":
    LEARNING_RATES = [0.01, 0.001, 0.0001]
    BATCH_SIZES = [10, 100, 1000]
    LOSSES = [SoftmaxLoss, LinearLeastSquaresLoss]

    for loss in LOSSES:
        params_combs = [(loss, lr, batch_size) for lr in LEARNING_RATES for batch_size in BATCH_SIZES]
        results = p_map(generate_sgd_per_hyperparams, *np.array(params_combs).T)
        best_idx = np.argmax([acc for acc, *_ in results])
        acc, accuracy_trains, accuracy_tests = results[best_idx]
        # show best fig
        fig, axs = plt.subplots(ncols=len(DATA_FILES))
        fig.suptitle(f'Accuracy, {loss.name}')
        fig.tight_layout(pad=3.0)
        for i, data_file in enumerate(DATA_FILES):
            axs[i].plot(accuracy_trains[i], label='train')
            axs[i].plot(accuracy_tests[i], label='test', alpha=0.6)
            axs[i].legend()
            axs[i].set_title(f"{data_file}\nlr={params_combs[best_idx][1]}\nbatch_size={params_combs[best_idx][2]}")
        fig.show()
        # print(f'last acc_train: {accuracy_train[-1]}, last acc_test: {accuracy_test[-1]}')
    plt.show()
