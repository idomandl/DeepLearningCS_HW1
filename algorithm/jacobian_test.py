import numpy as np
import matplotlib.pyplot as plt
from function.function import Function


class JacobianTest:
    def __init__(self, func: Function):
        self.func = func

    def __call__(self, X, Theta):
        epsilon = 0.01
        d_Theta = self.func.generate_tensor(*Theta.shape)
        d_Theta = d_Theta / np.linalg.norm(d_Theta)
        d_X = self.func.generate_tensor(*X.shape)
        d_X = d_X / np.linalg.norm(d_X)
        diffs_Theta, diffs_power_Theta, diffs_X, diffs_power_X, epsilons = [], [], [], [], []
        f = self.func(X, Theta)
        grad_Theta = self.func.grad_Theta(X, Theta)
        grad_X = self.func.grad_X(X, Theta)
        for _ in range(8):
            epsilon /= 2
            epsilons.append(epsilon)
            # change in Theta
            f_Theta_ed = self.func(X, Theta + epsilon * d_Theta)
            diffs_Theta.append(np.linalg.norm(f_Theta_ed - f))
            lol = grad_Theta@(epsilon*d_Theta).T
            print(f'shapes, lol: {lol.shape} ,f_Theta_ed: {f_Theta_ed.shape}, f: {f.shape}')
            diffs_power_Theta.append(np.linalg.norm(f_Theta_ed - f - epsilon*np.dot(grad_Theta.flatten(),d_Theta.flatten())))
            # change in X
            f_X_ed = self.func(X + epsilon * d_X, Theta)
            diffs_X.append(np.linalg.norm(f_X_ed - f))
            diffs_power_X.append(np.linalg.norm(f_X_ed - f - np.dot(grad_X.flatten(),d_X.flatten())*epsilon))

        fig, axs = plt.subplots(ncols=2)
        fig.suptitle(f"Jacobian Test, {self.func.name} Function")
        # plot change in Theta
        axs[0].semilogy(range(8), diffs_Theta, linestyle='--', marker='o', label="normal")
        axs[0].semilogy(range(8), diffs_power_Theta, linestyle='--', marker='o', label="power")
        axs[0].legend()
        axs[0].set_title("Gradient weights & biases")
        axs[0].set_xlabel('i')
        # plot change in X
        axs[1].semilogy(range(8), diffs_X, linestyle='--', marker='o', label="normal")
        axs[1].semilogy(range(8), diffs_power_X, linestyle='--', marker='o', label="power")
        axs[1].legend()
        axs[1].set_title(f"Gradient input")
        axs[1].set_xlabel('i')
        fig.show()
