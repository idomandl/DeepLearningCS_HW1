import numpy as np
from metric_function import MetricFunction


class SoftmaxAccuracy(MetricFunction):
    name = "SoftmaxAccuracy"

    def __init__(self, Theta=None, Y=None):
        super().__init__(Theta, Y)

    def __call__(self, X, Y=None, Theta=None):
        MetricFunction.__call__(self, X, Y, Theta)
        probs = self.softmax(X)
        preds = np.argmax(probs, axis=1)
        truths = np.argmax(self.Y, axis=1)
        return np.mean(preds == truths)

    def softmax(self, X):
        #        return np.exp(X @ self.Theta) / np.sum(np.exp(X @ self.Theta), axis=1, keepdims=True)
        max_element = (X @ self.Theta).max(axis=1, keepdims=True)
        return np.log(np.exp(X @ self.Theta - max_element) / sum(
            np.exp(X @ Theta_i.T[:, None] - max_element) for Theta_i in self.Theta.T))
