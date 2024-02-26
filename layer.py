import numpy as np
from function.function import Function
from algorithm.sgd import SGD

class Layer:
    def __init__(self, Theta_dim, activation: Function):
        self.activation = activation
        self.Theta = activation.generate_tensor(*Theta_dim) * 0.01

    def forward(self, X):
        self.X = X
        return self.activation(X, self.Theta)

    def backward(self, dA, optimizer: SGD):
        dTheta = self.activation.grad_Theta(self.X, self.Theta)
        dX = self.activation.grad_X(self.X, self.Theta)
        #print(f'dX = {dX.shape}, dA = {dA.shape}, dTheta = {dTheta.shape}')
        dThetaLoss = dTheta * dA.T
        #print(f'dThetaLoss = {dThetaLoss.shape}')
        self.Theta += optimizer.update_params(dThetaLoss, self)
        # dX = (batch_size, input), dA = (batch_size, output)
        return np.sum(dX.T, axis=1) * sum(dA)

# L - w(128,6),x(batch,128)
# A2 - w(16,128), x(batch,16), DA(batch,128)
# A1 - w(x.shape[1], 16), x(batch,x.shape[1])

    def backward_loss(self, X, Y, learning_rate):
        dL_Theta = self.activation.loss_grad_Theta(X, Y, self.Theta)
        dL_X = self.activation.loss_grad_X(X, Y, self.Theta)
        self.Theta -= learning_rate * dL_Theta
        return np.sum(dL_X, axis=0)

    def loss(self, X, Y):
        return self.activation.loss(X, Y, self.Theta)

    def metric(self, X, Y, metric):
        return metric(X, Y, self.Theta)
