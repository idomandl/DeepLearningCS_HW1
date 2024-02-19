import numpy as np
from function.function import Function


class Layer:
    def __init__(self, Theta_dim, activation: Function):
        self.activation = activation
        self.Theta = activation.generate_Theta(*Theta_dim) * 0.01

    def forward(self, X):
        self.X = X
        return self.activation(X @ self.Theta)

    def backward(self, dA, learning_rate=0.01):
        dTheta = self.activation.calc_grad_Theta(self.X, self.Theta)
        dX = self.activation.calc_grad_X(self.X, self.Theta)
        self.Theta -= learning_rate * dTheta @ dA
        return dX @ dA
