import matplotlib.pyplot as plt
from function.function import Function
from algorithm.jacobian_test import Jacobian_Test_Residual
from algorithm.jacobian_transposed_test import Jacobian_Transposed_Test_Residual
import numpy as np
from function.softmax import Softmax
from function.tanh import Tanh
from data_helper import DATA_FILES, get_data, read_file
from network.nn import NN
from algorithm.sgd_momentum import SGDMomentum
from network.residual_layer import Residual_Layer
from network.layer import Layer


DATA_FILE = DATA_FILES[0]

# ----- Jacobian Test -----
_, _, X_test, Y_test = read_file(DATA_FILE)
layer = Residual_Layer((X_test.shape[1], 16), Tanh())
Theta1 = Function().generate_tensor(X_test.shape[1], 16)
Theta2 = Function().generate_tensor(16, X_test.shape[1])
Jacobian_Test_Residual(layer)(X_test[np.random.randint(0, X_test.shape[0])].reshape(1, -1), Theta1, Theta2)
plt.show()

# ----- Jacobian Transposed Test -----
layer = Residual_Layer((10, 12), Tanh())
jac_t_test = Jacobian_Transposed_Test_Residual(layer)
results = [jac_t_test(np.random.rand(1, 10), np.random.rand(10, 12) * 0.01, np.random.rand(12, 10) * 0.01) for _ in range(1000)]
print(f'Jacobian Transposed Test: {max(results)}')

# ----- NN -----
LR = 0.001
STOP_CONDITION = 200
BATCH_SIZE = 10

X_train, Y_train, X_test, Y_test = get_data(DATA_FILE)
metrics = {'loss': Softmax().loss, 'accuracy': Softmax().accuracy}
layers = [Layer((X_train.shape[1], 10), Tanh()), Residual_Layer((10, 12), Tanh()), Layer((10, Y_train.shape[1]), Softmax())]
optimizer = SGDMomentum(metrics=metrics, lr=LR, stop_condition=STOP_CONDITION)
nn = NN(layers, optimizer, batch_size=BATCH_SIZE)
# train
metrics_results_train, metrics_results_test = nn.fit((X_train, Y_train), (X_test, Y_test))
# show loss
fig, axs = plt.subplots(ncols=len(metrics))
fig.suptitle(f'Metrics Results')
fig.tight_layout(pad=3.0)
for i, metric_name in enumerate(metrics.keys()):
    axs[i].plot([res[metric_name] for res in metrics_results_train], label=f'train {metric_name}')
    axs[i].plot([res[metric_name] for res in metrics_results_test], label=f'test {metric_name}', alpha=0.6)
    axs[i].set_title(metric_name)
    axs[i].legend()
    axs[i].set_xlabel('epoch')
plt.show()
