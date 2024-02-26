import matplotlib.pyplot as plt
from function.softmax import Softmax
from function.tanh import Tanh
from data_helper import DATA_FILES, get_data
from nn import NN
from algorithm.sgd import SGD
from algorithm.sgd_momentum import SGDMomentum

DATA_FILE = DATA_FILES[0]
LR = 0.05
BATCH_SIZE = 100
LOSS = Softmax()

X_train, Y_train, X_test, Y_test = get_data(DATA_FILE)
nn_dims = [
    X_train.shape[1],
    16,
    128,
    Y_train.shape[1],
]
nn_activations = [
    Tanh(),
    Tanh(),
    Softmax(),
]
metrics = {'loss': Softmax().loss, 'accuracy': Softmax().accuracy}
#optimizer = SGD(metrics, lr=LR, stop_condition=300)
optimizer = SGDMomentum(metrics, LR, 200)
nn = NN(nn_dims, nn_activations, optimizer, batch_size=BATCH_SIZE)
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
