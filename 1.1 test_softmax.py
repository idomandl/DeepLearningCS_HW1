from algorithm.gradient_test import GradientTest
from loss.softmax_loss import SoftmaxLoss
from loss.linear_least_squares_loss import LinearLeastSquaresLoss
import numpy as np
import matplotlib.pyplot as plt
from data_helper import DATA_FILES, read_file

def run_grad_test(loss, X):
    gradient_test = GradientTest(loss, loss.calc_grad, X.shape, loss.name)
    return gradient_test(X)

data_file = DATA_FILES[0]
_, _, X_test, Y_test = read_file(data_file)
Theta = np.random.randn(X_test.shape[1], Y_test.shape[1])
diffs, power_diffs, eps_is = run_grad_test(SoftmaxLoss(Theta, Y_test), X_test)
diffs, power_diffs, eps_is = run_grad_test(LinearLeastSquaresLoss(Theta, Y_test), X_test)
plt.show()
