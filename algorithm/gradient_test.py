import numpy as np
import matplotlib.pyplot as plt


class GradientTest:
    def __init__(self, func, grad_func, x_shape, func_name):
        self.func = func
        self.grad_func = grad_func
        self.x_shape = x_shape
        self.func_name = func_name

    def __call__(self, x):
        d = np.random.randn(*self.func.get_theta_shape(x))
        d = d / np.linalg.norm(d)
        epsilon = 0.01
        eps_i = epsilon
        f_x = self.func(x)
        diffs, diffs_power, eps_is = [], [], []
        grad_x = self.grad_func(x)
        Theta = self.func.Theta
        for _ in range(8):
            eps_i = eps_i / 2
            f_x_ed = self.func(x, Theta=Theta + eps_i * d)
            diffs.append(np.abs(f_x_ed - f_x))
            diffs_power.append(np.abs(f_x_ed - f_x - eps_i * np.dot(grad_x.flatten(), d.flatten())))
            eps_is.append(eps_i)

        fig, ax = plt.subplots()
        ax.semilogy(range(8), diffs, linestyle='--', marker='o', label="normal")
        ax.semilogy(range(8), diffs_power, linestyle='--', marker='o', label="power")
        ax.legend()
        ax.set_title(f"Gradient Test, {self.func_name}")
        ax.set_xlabel('i')
        fig.show()
        print(diffs, diffs_power, eps_is)
        return diffs, diffs_power, eps_is
