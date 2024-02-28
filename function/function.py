import numpy as np


class Function:
    name = "function"
    def __call__(self, X, Theta):
        raise NotImplementedError

    def grad_Theta(self, X, Theta):
        raise NotImplementedError

    def grad_X(self, X, Theta):
        raise NotImplementedError
    
    def grad(self, Z):
        raise NotImplementedError

    def loss(self, X, Y, Theta):
        raise NotImplementedError

    def loss_grad_Theta(self, X, Y, Theta):
        raise NotImplementedError

    def loss_grad_X(self, X, Y, Theta):
        raise NotImplementedError

    def accuracy(self, X, Y, Theta):
        probs = self(X, Theta)
        preds = np.argmax(probs, axis=1)
        truths = np.argmax(Y, axis=1)
        return np.mean(preds == truths)

    def get_Theta_shape(self, X, Y):
        return X.shape[1], Y.shape[1]

    def generate_tensor(self, *shape):
        return np.random.randn(*shape)
