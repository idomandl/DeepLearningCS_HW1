import numpy as np
from layer import Layer
from algorithm.sgd import SGD
from data_helper import split_to_batches, select_metric_sample


class NN:
    def __init__(self, network_dims, activations, optimizer: SGD, batch_size=100):
        if len(network_dims) != len(activations) + 1:
            raise ValueError("length of network_dims should be length of activations + 1")
        self.activations = activations
        self.layers = [Layer((network_dims[i], network_dims[i + 1]), activations[i]) for i in range(len(activations))]
        self.optimizer = optimizer
        self.batch_size = batch_size

    def forward(self, X):
        for layer in self.layers[:-1]:
            X = layer.forward(X)
        return X
    
    def predict(self, X):
        pass # TODO

    def backward(self, dL_X):
        if len(self.layers) == 1:
            return
        d_X = self.layers[-2].backward(dL_X, self.optimizer)
        for layer in reversed(self.layers[:-2]):
            d_X = layer.backward(d_X, self.optimizer)

    def fit(self, D_train, D_test):
        last_layer = self.layers[-1]
        # each iteration is an epoch
        while not self.optimizer.should_stop(last_layer.metric, self.forward, *D_test):
            # each iteration is a batch
            for X_batch, Y_batch in split_to_batches(*D_train, self.batch_size):
                # forward
                nn_output = self.forward(X_batch)
                # backward
                dL_X = last_layer.backward_loss(nn_output, Y_batch, self.optimizer.learning_rate)
                self.backward(dL_X)
                # update
                loss = last_layer.loss(nn_output, Y_batch)
                self.optimizer.update(loss, last_layer.metric, nn_output, Y_batch)
        return self.optimizer.metric_results_train, self.optimizer.metric_results_test
