from matplotlib import pyplot as plt

from gradient_test import GradientTest
from softmax_loss import SoftmaxLoss, SoftmaxLossLoop
from linear_least_squares_loss import LinearLeastSquaresLoss
import numpy as np

from scipy.optimize import check_grad
def main():
    theta = np.random.randn(2,2)
    y = np.random.randn(1, 2)
    softmax_loss = SoftmaxLossLoop(theta, y)
    gradient_test = GradientTest(softmax_loss, softmax_loss.calc_grad, (1, 2), "softmax loss")
    diffs, power_diffs, eps_is = gradient_test()
    print(eps_is)
    # linear_squares = LinearLeastSquaresLoss(theta, y)
    # gradient_test = GradientTest(linear_squares, linear_squares.calc_grad, (1,1), "linear squares loss")
    # gradient_test()
#     diffs = [0.07356278528598992,0.044236580085952326, 0.02398208690371284, 0.012456992667040367, 0.0063449836373159485, 0.0032016136446078036, 0.0016080872787913592, 0.0008058637535164337]
#     power_diffs = [0.029820749771815258,0.007455187442950262, 0.0018637968607393418, 0.00046594921518483545, 0.00011648730379754113,2.9121825948053015e-5, 7.2804564865691646e-6, 1.820114123418648e-6
# ]
#     e= 0.1
#     epis = [e*pow(0.5,i) for i in range(1,9)]
#     plt.plot(epis,diffs)
#     plt.plot(epis, power_diffs)
#     plt.legend(['Difference', 'Power'])
#     plt.show()

if __name__ == "__main__":
    main()
