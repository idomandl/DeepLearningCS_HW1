import numpy as np
from function.function import Function
from algorithm.sgd import SGD
from network.block import Block


class Residual_Layer(Block):
    def __init__(self, Theta_dim, activation: Function):
        self.activation = activation
        self.Theta1 = activation.generate_tensor(*Theta_dim) * 0.01
        self.Theta2 = activation.generate_tensor(*reversed(Theta_dim)) * 0.01

    def forward(self, X):
        self.X = X
        self.A = self.activation(X @ self.Theta1)
        return self.activation(self.A @ self.Theta2 + X)

    def backward(self, dA, optimizer: SGD):
        # input = output
        grad2 = self.activation.grad(self.A @ self.Theta2 + self.X)
        # (batch_size, hidden) = (batch_size, output) @ (output, hidden)
        dX2 = grad2 @ self.Theta2.T
        dTheta2 = self.A.T @ grad2
        dThetaLoss2 = dTheta2 * dA.T
        self.Theta2 += optimizer.update_params(dThetaLoss2, (self, 2))
        dA2 = np.sum(dX2.T, axis=1) * sum(dA)
        # ---------------------
        grad1 = self.activation.grad(self.X @ self.Theta1)
        # (batch_size, input) = (batch_size, hidden) @ (hidden, input)
        dX1 = grad1 @ self.Theta1.T
        dTheta1 = self.X.T @ grad1
        dThetaLoss1 = dTheta1 * dA2.T
        self.Theta1 += optimizer.update_params(dThetaLoss1, (self, 1))
        return np.sum(dX1.T, axis=1) * sum(dA2) + np.sum(grad2.T, axis=1) * sum(dA)

    def get_input_dim(self):
        return self.Theta1.shape[0]
    
    def get_output_dim(self):
        return self.Theta2.shape[1]
    