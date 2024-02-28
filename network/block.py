from algorithm.sgd import SGD


class Block:
    def forward(self, X):
        raise NotImplementedError

    def backward(self, dA, optimizer: SGD):
        raise NotImplementedError

    def backward_loss(self, X, Y, learning_rate):
        raise NotImplementedError

    def loss(self, X, Y):
        raise NotImplementedError

    def metric(self, X, Y, metric):
        raise NotImplementedError

    def get_input_dim(self):
        raise NotImplementedError
    
    def get_output_dim(self):
        raise NotImplementedError
