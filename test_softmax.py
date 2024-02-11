from algorithm.gradient_test import GradientTest
from loss.softmax_loss import SoftmaxLoss
from loss.linear_least_squares_loss import LinearLeastSquaresLoss
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def run_grad_test(loss, X_train):
    gradient_test = GradientTest(loss, loss.calc_grad, X_train.shape, loss.name)
    return gradient_test(X_train)

mat_contents = sio.loadmat(f'data/GMMData.mat')
Y_train = mat_contents['Ct'].T
X_train = mat_contents['Yt'].T
Theta = np.random.randn(X_train.shape[1], Y_train.shape[1])  # np.array([[0.7, 0.2]])
# y = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0]])  # shape=(5,3)#np.array([[1,0]])
diffs, power_diffs, eps_is = run_grad_test(SoftmaxLoss(Theta, Y_train), X_train)
diffs, power_diffs, eps_is = run_grad_test(LinearLeastSquaresLoss(Theta, Y_train), X_train)
plt.show()
