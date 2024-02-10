import numpy as np
class LossFunction:
    def __init__(self, Theta, Y):
        self.Theta = Theta
        self.Y = Y

    def set_Theta(self, Theta):
        self.Theta = Theta

    def set_Y(self, Y):
        self.Y = Y

    def __call__(self, X):
        pass

    def calc_grad(self, X):
        pass

    def get_theta_shape(self, X):
        return X.shape[1], self.Y.shape[1]

