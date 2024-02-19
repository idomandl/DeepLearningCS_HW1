import numpy as np
import matplotlib.pyplot as plt
from function.function import Function


class GradientTest:
    def __init__(self, loss: Function):
        self.loss = loss

    def __call__(self, X, Y, Theta):
        epsilon = 0.01
        d = self.loss.generate_Theta(*self.loss.get_Theta_shape(X, Y))
        print(f'{d.shape=}')
        d = d / np.linalg.norm(d)
        diffs, diffs_power, epsilons = [], [], []
        f_x = self.loss.loss(X, Y, Theta)
        grad_x = self.loss.loss_grad_Theta(X, Y, Theta)
        for _ in range(8):
            epsilon /= 2
            f_x_ed = self.loss.loss(X, Y, Theta + epsilon * d)
            diffs.append(np.abs(f_x_ed - f_x))
            diffs_power.append(np.abs(f_x_ed - f_x - epsilon * np.dot(grad_x.flatten(), d.flatten())))
            epsilons.append(epsilon)

        fig, ax = plt.subplots()
        ax.semilogy(range(8), diffs, linestyle='--', marker='o', label="normal")
        ax.semilogy(range(8), diffs_power, linestyle='--', marker='o', label="power")
        ax.legend()
        ax.set_title(f"Gradient Test, {self.loss.name} Loss")
        ax.set_xlabel('i')
        fig.show()
        print(diffs, diffs_power, epsilons)
        return diffs, diffs_power, epsilons
