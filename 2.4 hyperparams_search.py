import matplotlib.pyplot as plt
import numpy as np
from function.softmax import Softmax
from function.tanh import Tanh
from data_helper import DATA_FILES, get_data
from network.nn import NN
from algorithm.sgd_momentum import SGDMomentum
from network.residual_layer import Residual_Layer
from network.layer import Layer
from p_tqdm import p_map
import pickle

STOP_CONDITION = 200
metrics = {'loss': Softmax().loss, 'accuracy': Softmax().accuracy}

def create_layers(depth, layer_type, X_train, Y_train):
    first = [Layer((X_train.shape[1], 16), Tanh())]
    last = [Layer((16, Y_train.shape[1]), Softmax())]
    return first + [layer_type((16, 16), Tanh()) for _ in range(depth)] + last

def create_nn(layers, lr, batch_size):
    optimizer = SGDMomentum(metrics=metrics, lr=lr, stop_condition=STOP_CONDITION, log=False)
    return NN(layers, optimizer, batch_size)

def run_nn_per_hyperparams(layer_type, depth, lr, batch_size, data_file):
    X_train, Y_train, X_test, Y_test = get_data(data_file)
    layers = create_layers(depth, layer_type, X_train, Y_train)
    nn = create_nn(layers, lr, batch_size)
    metric_results_train, metric_results_test = nn.fit((X_train, Y_train), (X_test, Y_test))
    return nn, metric_results_train, metric_results_test


if __name__ == "__main__":

    LEARNING_RATES = [0.01, 0.001, 0.0001]
    BATCH_SIZES = [10, 100, 1000]

    fig, axs = plt.subplots(nrows=2, ncols=len(DATA_FILES))
    fig.tight_layout(pad=3.0)
    
    for i, data_file in enumerate(DATA_FILES):
        X_train, Y_train, X_test, Y_test = get_data(data_file)
        
        params_combs = [[layer_type, depth, lr, batch_size, data_file] for lr in LEARNING_RATES for batch_size in BATCH_SIZES for depth in range(1, 4) for layer_type in [Layer, Residual_Layer]]
        results = p_map(run_nn_per_hyperparams, *np.array(params_combs).T) # parallel computing
        best_idx = np.argmax([metric_test[-1]['accuracy'] for _, _, metric_test in results])
        best_nn, metric_train, metric_test = results[best_idx]
        pickle.dump(results, open(f'results/2.4 {data_file}.pkl', 'wb'))

        # show accuracy of the best hyperparameters
        axs[0,i].plot([x['accuracy'] for x in metric_train], label='train accuracy')
        axs[0,i].plot([x['accuracy'] for x in metric_test], label='test accuracy', alpha=0.6)
        axs[0,i].legend()
        axs[0,i].set_title(f"lr={best_nn.optimizer.learning_rate}\nbatch_size={best_nn.batch_size}\nlayer_type={best_nn.blocks[1].__class__.__name__}\nlayer_depth={len(best_nn.blocks) - 2}")
        axs[0,i].set_xlabel('epoch')
        axs[1,i].plot([x['loss'] for x in metric_train], label='train loss')
        axs[1,i].plot([x['loss'] for x in metric_test], label='test loss', alpha=0.6)
        axs[1,i].legend()
        axs[1,i].set_title(f"{data_file}")
        axs[1,i].set_xlabel('epoch')
    plt.show()
