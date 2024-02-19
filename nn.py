from layer import Layer


class NN:
    def __init__(self, network_dims, activations, sgd_fn):
        if len(network_dims) != len(activations) + 1:
            raise ValueError("length of network_dims should be length of activations + 1")
        self.activations = activations
        self.layers = [Layer((network_dims[i], network_dims[i + 1]), activations[i]) for i in range(len(activations))]
        self.sgd_fn = sgd_fn

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dL, learning_rate=0.01):
        for layer in reversed(self.layers):
            dL = layer.backward(dL, learning_rate)
