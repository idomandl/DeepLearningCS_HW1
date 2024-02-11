import numpy as np
from metric_function import MetricFunction


class SoftmaxLoss(MetricFunction):
    name = "SoftmaxLoss"

    def __init__(self, Theta, Y):
        super().__init__(Theta, Y)

    def __call__(self, X, Y=None, Theta=None):
        MetricFunction.__call__(self, X, Y, Theta)
        max_element = (X @ self.Theta).max(axis=1, keepdims=True)
        # sum((batch_size, m) * (batch_size, m)) / batch_size
        return -(np.sum(self.Y * np.log(np.exp(X @ self.Theta - max_element) / sum(
            np.exp(X @ Theta_i.T[:, None] - max_element) for Theta_i in self.Theta.T))) / X.shape[0])

    def calc_grad(self, X, Y=None, Theta=None):
        MetricFunction.calc_grad(self, X, Y, Theta)
        max_element = (X @ self.Theta).max(axis=1, keepdims=True)
        # (n, batch_size) @ (batch_size, m) = (n, m)
        return X.T @ (np.exp(X @ self.Theta - max_element) / sum(
            np.exp(X @ Theta_i.T[:, None] - max_element) for Theta_i in self.Theta.T) - self.Y) / X.shape[0]
