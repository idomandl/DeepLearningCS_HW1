from algorithm.jacobian_test import Jacobian_Test
from function.tanh import Tanh
from function.function import Function
import matplotlib.pyplot as plt
from data_helper import DATA_FILES, read_file


def run_jacobian_test(fn, X, Theta):
    Jacobian_Test(fn)(X, Theta)


data_file = DATA_FILES[0]
_, _, X_test, Y_test = read_file(data_file)
Theta = Function().generate_tensor(*Function().get_Theta_shape(X_test, Y_test))
run_jacobian_test(Tanh(), X_test, Theta)
plt.show()
