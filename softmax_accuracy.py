import numpy as np
class SoftmaxAccuracy:
    name = "SoftmaxAccuracy"
    def __init__(self, Theta, Y):
        self.Theta = Theta
        self.Y = Y

    def set_Theta(self, Theta):
        self.Theta = Theta

    def set_Y(self, Y):
        self.Y = Y
    def __call__(self, X):
        proba = self.softmax(X)
        preds = np.argmax(proba, axis=1)
        truths = np.argmax(self.Y, axis=1)
        return np.mean(preds == truths)
    def softmax(self, X):
#        return np.exp(X @ self.Theta) / np.sum(np.exp(X @ self.Theta), axis=1, keepdims=True)
        max_element = (X @ self.Theta).max(axis=1, keepdims=True)
        return np.log(np.exp(X @ self.Theta - max_element) / sum(
            np.exp(X @ Theta_i.T[:, None] - max_element) for Theta_i in self.Theta.T))