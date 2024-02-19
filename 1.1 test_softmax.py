from algorithm.gradient_test import GradientTest
from function.softmax import Softmax
from function.linear_least_squares import LinearLeastSquares
from function.function import Function
import matplotlib.pyplot as plt
from data_helper import DATA_FILES, read_file

def run_grad_test(fn, X, Y, Theta):
    return GradientTest(fn)(X, Y, Theta)

data_file = DATA_FILES[0]
_, _, X_test, Y_test = read_file(data_file)
Theta = Function().generate_Theta(*Function().get_Theta_shape(X_test, Y_test))
run_grad_test(Softmax(), X_test, Y_test, Theta)
run_grad_test(LinearLeastSquares(), X_test, Y_test, Theta)
plt.show()
