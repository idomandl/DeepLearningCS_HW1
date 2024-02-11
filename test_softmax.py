from matplotlib import pyplot as plt

from gradient_test import GradientTest
from softmax_loss import SoftmaxLoss, SoftmaxLossLoop
from linear_least_squares_loss import LinearLeastSquaresLoss
import numpy as np
import scipy.io as sio


from scipy.optimize import check_grad
def main():
    mat_contents = sio.loadmat(f'GMMData.mat')
    Y_train = mat_contents['Ct'].T
    X_train = mat_contents['Yt'].T
    theta = np.random.randn(X_train.shape[1],Y_train.shape[1])#np.array([[0.7, 0.2]])
    y = np.array([[0,0,1],[0,0,1],[0,1,0],[1,0,0],[0,1,0]])#shape=(5,3)#np.array([[1,0]])
    softmax_loss = SoftmaxLoss(theta, Y_train)
    gradient_test = GradientTest(softmax_loss, softmax_loss.calc_grad, X_train.shape, "softmax loss")
    diffs, power_diffs, eps_is = gradient_test(X_train)
    linear_loss = LinearLeastSquaresLoss(theta, Y_train)
    gradient_test = GradientTest(linear_loss, linear_loss.calc_grad, X_train.shape, "linear loss")
    diffs, power_diffs, eps_is = gradient_test(X_train)

if __name__ == "__main__":
    main()
