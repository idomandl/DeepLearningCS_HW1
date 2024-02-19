import numpy as np
import matplotlib.pyplot as plt
from function.softmax import Softmax
from p_tqdm import p_map
from data_helper import DATA_FILES
from algo_helper import run_sgd
import pickle

def run_sgd_per_hyperparams(loss, lr, batch_size):
    acc_tot = 0
    accuracy_trains, accuracy_tests = [], []
    for data_file in DATA_FILES:
        # print(f'Processing {data_file}')
        _, accuracy_train, accuracy_test = run_sgd(data_file, loss, metric_fn=Softmax().accuracy, lr=lr, batch_size=batch_size)
        acc_tot += np.median(accuracy_test[-10:])
        accuracy_trains.append(accuracy_train)
        accuracy_tests.append(accuracy_test)
    return acc_tot, accuracy_trains, accuracy_tests


if __name__ == "__main__":
    LEARNING_RATES = [0.01, 0.001, 0.0001]
    BATCH_SIZES = [10, 100, 1000]
    LOSS = Softmax()

    params_combs = [(LOSS, lr, batch_size) for lr in LEARNING_RATES for batch_size in BATCH_SIZES]
    results = p_map(run_sgd_per_hyperparams, *np.array(params_combs).T) # parallel computing
    best_idx = np.argmax([acc for acc, *_ in results])
    acc, accuracy_trains, accuracy_tests = results[best_idx]
    pickle.dump(results, open('results/1.3.pkl', 'wb'))
    # show accuracy of the best hyperparameters
    fig, axs = plt.subplots(ncols=len(DATA_FILES))
    fig.suptitle(f'Accuracy, {LOSS.name} Loss')
    fig.tight_layout(pad=3.0)
    for i, data_file in enumerate(DATA_FILES):
        axs[i].plot(accuracy_trains[i], label='train accuracy')
        axs[i].plot(accuracy_tests[i], label='test accuracy', alpha=0.6)
        axs[i].legend()
        axs[i].set_title(f"{data_file}\nlr={params_combs[best_idx][1]}\nbatch_size={params_combs[best_idx][2]}")
        axs[i].set_xlabel('epoch')
    plt.show()
