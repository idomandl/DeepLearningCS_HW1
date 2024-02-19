import numpy as np
from function.function import Function


class TanhActivation(Function):
    name = "tanh"

    def __call__(self, X, Theta):
        return np.tanh(X @ Theta)

    def grad_X(self, X, Theta):
        return (1 - np.tanh(X @ Theta) ** 2) @ Theta.T

    def grad_Theta(self, X, Theta):
        return X.T @ (1 - np.tanh(X @ Theta) ** 2)
