from data_helper import select_metric_sample
from tqdm import tqdm


class SGD:
    def __init__(self, metrics, lr=0.01, stop_condition=200, metric_sample_percentage=0.3, log=True):
        self.metrics = metrics
        self.learning_rate = lr
        self.stop_condition = stop_condition
        self.metric_sample_percentage = metric_sample_percentage
        self.log = log
        self.compare_window = 50
        self.losses = []
        self.metric_results_train = []
        self.metric_results_test = []
        self.batch_results_train = []
        self.epoch = 0
        self.epoch_bar = tqdm(total = stop_condition)


    def should_stop(self, calc_metric, forward, X, Y):
        self.epoch += 1
        self.epoch_bar.update()
        if self.epoch == 1:
            return False
        X_sample, Y_sample = select_metric_sample(X, Y, self.metric_sample_percentage)
        self.metric_results_test.append({
            name: calc_metric(forward(X_sample), Y_sample, fn) for name, fn in self.metrics.items()
        })
        self.metric_results_train.append({
            name: sum([res[name] for res in self.batch_results_train]) / len(self.batch_results_train) for name in self.metrics.keys()
        })
        if self.log and self.epoch % 50 == 1:
            print(f'train = {self.metric_results_train[-1]} , test = {self.metric_results_test[-1]}')
        return self.epoch > self.stop_condition
        # if len(self.losses) < self.compare_window:
        #     return False
        # return max(abs(self.losses[-self.compare_window:] - self.losses[-1])) < self.stop_condition

    def update_params(self, Theta_grads, *args):
        return - self.learning_rate * Theta_grads

    def update(self, loss, calc_metric, X, Y):
        self.losses.append(loss)
        sample = select_metric_sample(X, Y, self.metric_sample_percentage)
        self.batch_results_train.append({
            name: calc_metric(*sample, fn) for name, fn in self.metrics.items()
        })
