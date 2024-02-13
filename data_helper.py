import numpy as np
import scipy.io as sio

DATA_FILES = ['GMMData', 'PeaksData', 'SwissRollData']

def shuffle_data(X, Y):
    indices = np.random.permutation(X.shape[0])
    return X[indices], Y[indices]

def add_bias(X):
    return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

def read_file(data_file):
    mat_contents = sio.loadmat(f'data/{data_file}.mat')
    Y_train = mat_contents['Ct'].T
    X_train = mat_contents['Yt'].T
    Y_test = mat_contents['Cv'].T
    X_test = mat_contents['Yv'].T
    return X_train, Y_train, X_test, Y_test

def get_data(data_file):
    X_train, Y_train, X_test, Y_test = read_file(data_file)
    # Shuffle the data
    X_train, Y_train = shuffle_data(X_train, Y_train)
    X_test, Y_test = shuffle_data(X_test, Y_test)
    # Add bias term
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)
    return X_train, Y_train, X_test, Y_test
