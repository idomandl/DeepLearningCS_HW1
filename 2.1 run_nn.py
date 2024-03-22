import matplotlib.pyplot as plt
from function.softmax import Softmax
from function.tanh import Tanh
from data_helper import DATA_FILES, get_data, read_file
from network.nn import NN
from function.function import Function
from algorithm.sgd_momentum import SGDMomentum
from algorithm.jacobian_test import Jacobian_Test
from algorithm.jacobian_transposed_test import Jacobian_Transposed_Test
from network.layer import Layer
import numpy as np


DATA_FILE = DATA_FILES[0]

# ----- Jacobian Test -----
_, _, X_test, Y_test = read_file(DATA_FILE)
Theta = Function().generate_tensor(*Function().get_Theta_shape(X_test, Y_test))
Jacobian_Test(Tanh())(X_test[np.random.randint(0, X_test.shape[0])].reshape(1, -1), Theta)
plt.show()

# ----- Jacobian Transposed Test -----
jac_t_test = Jacobian_Transposed_Test(Tanh())
results = [jac_t_test(np.random.rand(1, 10), np.random.rand(10, 5) * 0.01) for _ in range(1000)]
print(f'Jacobian Transposed Test: {max(results)}')

# ----- NN -----
LR = 0.05
STOP_CONDITION = 200
BATCH_SIZE = 100

X_train, Y_train, X_test, Y_test = get_data(DATA_FILE)
layers = [Layer((X_train.shape[1], 16), Tanh()), Layer((16, 128), Tanh()), Layer((128, Y_train.shape[1]), Softmax())]
metrics = {'loss': Softmax().loss, 'accuracy': Softmax().accuracy}
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
