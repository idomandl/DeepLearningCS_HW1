import matplotlib.pyplot as plt
from function.softmax import Softmax
from function.tanh import Tanh
from data_helper import DATA_FILES, get_data
from network.nn import NN
from algorithm.sgd_momentum import SGDMomentum
from network.residual_layer import Residual_Layer
from network.layer import Layer
from algorithm.gradient_test import GradientTestNetwork
import numpy as np


data_file = DATA_FILES[0]
LR = 0.05
STOP_CONDITION = 200
BATCH_SIZE = 1

X_train, Y_train, X_test, Y_test = get_data(data_file)
metrics = {'loss': Softmax().loss, 'accuracy': Softmax().accuracy}
# layers = [Layer((X_train.shape[1], 10), Tanh()), Residual_Layer((10, 12), Tanh()), Layer((10, Y_train.shape[1]), Softmax())]
layers = [Layer((X_train.shape[1], Y_train.shape[1]), Softmax())]
optimizer = SGDMomentum(metrics=metrics, lr=LR, stop_condition=STOP_CONDITION)
nn = NN(layers, optimizer, batch_size=BATCH_SIZE)
idx = np.random.randint(0, X_test.shape[0])
GradientTestNetwork(nn)(X_test[idx].reshape(1, -1), Y_test[idx].reshape(1, -1))
plt.show()
