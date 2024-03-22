from network.block import Block
from algorithm.sgd import SGD
from data_helper import split_to_batches


class NN:
    name="NN"

    def __init__(self, blocks: list[Block], optimizer: SGD, batch_size=100):
        # if len(network_dims) != len(activations) + 1:
        #     raise ValueError("length of network_dims should be length of activations + 1")
        # self.activations = activations
        # self.layers = [Layer((network_dims[i], network_dims[i + 1]), activations[i]) for i in range(len(activations))]
        self.blocks = []
        for block in blocks:
            self.add_block(block)
        self.optimizer = optimizer
        self.batch_size = batch_size
    
    def add_block(self, block):
        if len(self.blocks) > 0:
            cur_input_dim = block.get_input_dim()
            prev_output_dim = self.blocks[-1].get_output_dim()
            if prev_output_dim != cur_input_dim:
                raise ValueError(f"Block {len(self.blocks)} output dim {prev_output_dim} "
                                f"should be equal to block {len(self.blocks) + 1} input dim {cur_input_dim}")
        self.blocks.append(block)


    def forward(self, X):
        for block in self.blocks[:-1]:
            X = block.forward(X)
        return X
    
    def predict(self, X):
        nn_output = self.forward(X)
        return self.blocks[-1].activation(nn_output, self.blocks[-1].Theta)

    def backward(self, dL_X, is_training=True):
        if len(self.blocks) == 1:
            return
        d_X = self.blocks[-2].backward(dL_X, self.optimizer, is_training=is_training)
        for block in reversed(self.blocks[:-2]):
            d_X = block.backward(d_X, self.optimizer, is_training=is_training)
        return d_X
    
    def loss(self, X, Y):
        nn_output = self.forward(X)
        return self.blocks[-1].loss(nn_output, Y)
    
    def grad_X(self, X, Y):
        nn_output = self.forward(X)
        dL_X = self.blocks[-1].backward_loss(nn_output, Y, self.optimizer.learning_rate, is_training=False)
        grad_X = self.backward(dL_X, is_training=False)
        return grad_X

    def fit(self, D_train, D_test):
        last_block = self.blocks[-1]
        # each iteration is an epoch
        while not self.optimizer.should_stop(last_block.metric, self.forward, *D_test):
            # each iteration is a batch
            for X_batch, Y_batch in split_to_batches(*D_train, self.batch_size):
                # forward
                nn_output = self.forward(X_batch)
                # backward
                dL_X = last_block.backward_loss(nn_output, Y_batch, self.optimizer.learning_rate)
                self.backward(dL_X)
                # update
                loss = last_block.loss(nn_output, Y_batch)
                self.optimizer.update(loss, last_block.metric, nn_output, Y_batch)
        return self.optimizer.metric_results_train, self.optimizer.metric_results_test
