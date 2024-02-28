import numpy as np
from function.function import Function


class Tanh(Function):
    name = "tanh"

    def __call__(self, Z):
        return np.tanh(Z)

    def grad_X(self, X, Theta):
        return self.grad(X @ Theta) @ Theta.T

    def grad_Theta(self, X, Theta):
        return X.T @ self.grad(X @ Theta)

    def grad(self, Z):
        return 1 - (np.tanh(Z)**2)
