import numpy as np
import matplotlib.pyplot as plt
from function.function import Function
from network.residual_layer import Residual_Layer


class Jacobian_Test:
    def __init__(self, func: Function):
        self.func = func

    def __call__(self, X, Theta):
        epsilon = 0.01
        d_Theta = self.func.generate_tensor(*Theta.shape)
        d_Theta = d_Theta / np.linalg.norm(d_Theta)
        d_X = self.func.generate_tensor(*X.shape)
        d_X = d_X / np.linalg.norm(d_X)
        d_Y = self.func.generate_tensor(1, Theta.shape[1])
        d_Y = d_Y / np.linalg.norm(d_Y)
        diffs_Theta, diffs_power_Theta, diffs_X, diffs_power_X, epsilons = [], [], [], [], []
        f = self.func(X @ Theta)
        grad_Theta = self.func.grad_Theta(X, Theta)
        grad = self.func.grad(X @ Theta)
        grad_U = d_Y * grad
        grad_UX = grad_U @ Theta.T
        for _ in range(8):
            epsilon /= 2
            epsilons.append(epsilon)
            # change in Theta
            f_Theta_ed = self.func(X @ (Theta + epsilon * d_Theta))
            diffs_Theta.append(np.linalg.norm(f_Theta_ed - f))
            diffs_power_Theta.append(np.linalg.norm(f_Theta_ed - f - epsilon * np.sum(d_Theta * grad_Theta, axis=0)))
            # change in X
            f_X_ed = self.func((X + epsilon * d_X) @ Theta)
            diffs_X.append(abs(np.vdot(f_X_ed, d_Y) - np.vdot(f, d_Y)))
            diffs_power_X.append(abs(np.vdot(f_X_ed, d_Y) - np.vdot(f, d_Y) - np.vdot(grad_UX, epsilon * d_X)))

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

class Jacobian_Test_Residual:
    def __init__(self, block: Residual_Layer):
        self.block = block

    def __call__(self, X, Theta1, Theta2):
        epsilon = 0.01
        d_Theta = Function().generate_tensor(*Theta1.shape)
        d_Theta = d_Theta / np.linalg.norm(d_Theta)
        d_X = Function().generate_tensor(*X.shape)
        d_X = d_X / np.linalg.norm(d_X)
        u_Theta1 = Function().generate_tensor(1, Theta1.shape[1])
        u_Theta1 = u_Theta1 / np.linalg.norm(u_Theta1)
        u_Theta2 = Function().generate_tensor(1, Theta2.shape[1])
        u_Theta2 = u_Theta2 / np.linalg.norm(u_Theta2)
        diffs_Theta, diffs_power_Theta, diffs_X, diffs_power_X, epsilons = [], [], [], [], []
        self.block.Theta1 = Theta1
        self.block.Theta2 = Theta2
        f = self.block.forward(X)
        grad_X, grad_Theta1, grad_Theta2, grad1, grad2 = self.block.grad_test()
        grad_UX = (grad1 * sum((u_Theta2 * grad2) @ Theta2.T)) @ Theta1.T + (u_Theta2 * grad2)
        for _ in range(8):
            epsilon /= 2
            epsilons.append(epsilon)
            # change in Theta2
            self.block.Theta1 = Theta1
            self.block.Theta2 = Theta2 + epsilon * d_Theta.T
            f_Theta_ed = self.block.forward_test(X, u_Theta1, u_Theta2)
            diffs_Theta.append(np.linalg.norm(f_Theta_ed - f))
            # print(f'{f.shape=}, {grad_X.shape=}, {grad_Theta.shape=}, {d_Theta.shape=}, {d_X.shape=}')
            diffs_power_Theta.append(np.linalg.norm(f_Theta_ed - f - epsilon * np.sum(d_Theta.T * grad_Theta2, axis=0)))
            # change in Theta1
            # self.block.Theta1 = Theta1 + epsilon * d_Theta
            # self.block.Theta2 = Theta2
            # f_Theta_ed = self.block.forward(X)
            # diffs_Theta.append(np.linalg.norm(f_Theta_ed - f))
            # print(f'{f.shape=}, {grad_X.shape=}, {grad_Theta.shape=}, {d_Theta.shape=}, {d_X.shape=}')
            # diffs_power_Theta.append(np.linalg.norm(f_Theta_ed - f - epsilon * np.sum(d_Theta * grad_Theta1, axis=1).T))
            # change in X
            self.block.Theta1 = Theta1
            self.block.Theta2 = Theta2
            f_X_ed = self.block.forward_test(X + epsilon * d_X, u_Theta1, u_Theta2)
            diffs_X.append(abs(f_X_ed - f))
            diffs_power_X.append(abs(f_X_ed - f - np.vdot(grad_UX, epsilon * d_X)))

        fig, axs = plt.subplots(ncols=1)
        fig.suptitle(f"Jacobian Test, {self.block.name} Block")
        # plot change in Theta
        axs.semilogy(range(8), diffs_Theta, linestyle='--', marker='o', label="normal")
        axs.semilogy(range(8), diffs_power_Theta, linestyle='--', marker='o', label="power")
        axs.legend()
        axs.set_title("Gradient weights & biases")
        axs.set_xlabel('i')
        # plot change in X
        # axs[1].semilogy(range(8), diffs_X, linestyle='--', marker='o', label="normal")
        # axs[1].semilogy(range(8), diffs_power_X, linestyle='--', marker='o', label="power")
        # axs[1].legend()
        # axs[1].set_title(f"Gradient input")
        # axs[1].set_xlabel('i')
        fig.show()
