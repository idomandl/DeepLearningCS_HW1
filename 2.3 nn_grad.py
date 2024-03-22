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

metrics = {'loss': Softmax().loss, 'accuracy': Softmax().accuracy}
layers = [Layer((6, 10), Tanh()), Residual_Layer((10, 12), Tanh()), Layer((10, 5), Softmax())]
optimizer = SGDMomentum(metrics=metrics, lr=LR, stop_condition=STOP_CONDITION)
nn = NN(layers, optimizer, batch_size=BATCH_SIZE)
GradientTestNetwork(nn)()
plt.show()
