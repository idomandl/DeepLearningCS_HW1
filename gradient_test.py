import numpy as np
import matplotlib.pyplot as plt


class GradientTest:
    def __init__(self, func, grad_func, x_shape, func_name):
        self.func = func
        self.grad_func = grad_func
        self.x_shape = x_shape
        self.func_name = func_name

    def __call__(self):
        x = np.random.randn(*self.x_shape)
        d = np.random.randn(*self.func.get_theta_shape(x))
        d = d/np.linalg.norm(d,2)
        epsilon = 0.01
        eps_i = epsilon
        f_x = self.func(x)
        diffs = []
        diffs_power = []
        eps_is = []
        grad_x = self.grad_func(x)
        Theta = self.func.Theta
        dot = np.linalg.norm(d.T @ grad_x)
        for i in range(8):
            eps_i = eps_i / 2
            self.func.set_Theta(Theta + eps_i * d)
            f_x_ed = self.func(x)
            diffs.append(abs(f_x_ed - f_x))
            diffs_power.append(abs(f_x_ed - f_x - eps_i * dot))
            eps_is.append(eps_i)

        plt.semilogy(range(8), diffs, label="normal")
        plt.semilogy(range(8), diffs_power, label="power")
        plt.legend()
        plt.title(f"Gradient Test, {self.func_name}")
        plt.show()
        return diffs, diffs_power, eps_is


