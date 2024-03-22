import numpy as np
import matplotlib.pyplot as plt
from function.function import Function
from network.nn import NN
from data_helper import add_bias


class GradientTest:
    def __init__(self, loss: Function):
        self.loss = loss

    def __call__(self, X, Y, Theta):
        epsilon = 0.01
        d_Theta = self.loss.generate_tensor(*Theta.shape)
        d_Theta = d_Theta / np.linalg.norm(d_Theta)
        d_X = self.loss.generate_tensor(*X.shape)
        d_X = d_X / np.linalg.norm(d_X)
        diffs_Theta, diffs_power_Theta, diffs_X, diffs_power_X, epsilons = [], [], [], [], []
        f = self.loss.loss(X, Y, Theta)
        grad_Theta = self.loss.loss_grad_Theta(X, Y, Theta)
        grad_X = self.loss.loss_grad_X(X, Y, Theta)
        for _ in range(8):
            epsilon /= 2
            epsilons.append(epsilon)
            # change in Theta
            f_Theta_ed = self.loss.loss(X, Y, Theta + epsilon * d_Theta)
            diffs_Theta.append(np.abs(f_Theta_ed - f))
            diffs_power_Theta.append(np.abs(f_Theta_ed - f - epsilon * np.dot(grad_Theta.flatten(), d_Theta.flatten())))
            # change in X
            f_X_ed = self.loss.loss(X + epsilon * d_X, Y, Theta)
            diffs_X.append(np.abs(f_X_ed - f))
            diffs_power_X.append(np.abs(f_X_ed - f - epsilon * np.dot(grad_X.flatten(), d_X.flatten())))

        fig, axs = plt.subplots(ncols=2)
        fig.suptitle(f"Gradient Test, {self.loss.name} Loss")
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


class GradientTestNetwork:
    def __init__(self, nn: NN):
        self.nn = nn

    def __call__(self, X, Y):
        epsilon = 0.01
        # d_Theta = Function().generate_tensor(*Theta.shape)
        # d_Theta = d_Theta / np.linalg.norm(d_Theta)
        d_X = np.concatenate((Function().generate_tensor(X.shape[0], X.shape[1] - 1), np.zeros((X.shape[0], 1))), axis=1)
        d_X = d_X / np.linalg.norm(d_X)
        diffs_Theta, diffs_power_Theta, diffs_X, diffs_power_X, epsilons = [], [], [], [], []
        f = self.nn.loss(X, Y)
        # grad_Theta = self.nn.loss_grad_Theta(X, Y, Theta)
        grad_X = self.nn.grad_X(X, Y)
        for _ in range(8):
            epsilon /= 2
            epsilons.append(epsilon)
            # change in Theta
            # f_Theta_ed = self.nn.loss(X, Y, Theta + epsilon * d_Theta)
            # diffs_Theta.append(np.abs(f_Theta_ed - f))
            # diffs_power_Theta.append(np.abs(f_Theta_ed - f - epsilon * np.dot(grad_Theta.flatten(), d_Theta.flatten())))
            # change in X
            f_X_ed = self.nn.loss(X + epsilon * d_X, Y)
            diffs_X.append(np.abs(f_X_ed - f))
            diffs_power_X.append(np.abs(f_X_ed - f - epsilon * np.dot(grad_X.flatten(), d_X.flatten())))

        fig, axs = plt.subplots(ncols=2)
        fig.suptitle(f"Gradient Test, {self.nn.name} Loss")
        # plot change in Theta
        # axs[0].semilogy(range(8), diffs_Theta, linestyle='--', marker='o', label="normal")
        # axs[0].semilogy(range(8), diffs_power_Theta, linestyle='--', marker='o', label="power")
        # axs[0].legend()
        # axs[0].set_title("Gradient weights & biases")
        # axs[0].set_xlabel('i')
        # plot change in X
        axs[1].semilogy(range(8), diffs_X, linestyle='--', marker='o', label="normal")
        axs[1].semilogy(range(8), diffs_power_X, linestyle='--', marker='o', label="power")
        axs[1].legend()
        axs[1].set_title(f"Gradient input")
        axs[1].set_xlabel('i')
        fig.show()
