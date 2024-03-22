import numpy as np
import matplotlib.pyplot as plt
from function.function import Function
from network.nn import NN
from data_helper import add_bias
from network.residual_layer import Residual_Layer


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

    def __call__(self):
        epsilon = 0.01
        # d_Theta = Function().generate_tensor(*Theta.shape)
        # d_Theta = d_Theta / np.linalg.norm(d_Theta)
        diffs_Theta, diffs_power_Theta, diffs_X, diffs_power_X, epsilons = [[] for _ in range(len(self.nn.blocks)-1)], [[] for _ in range(len(self.nn.blocks)-1)], [[] for _ in range(len(self.nn.blocks)-1)], [[] for _ in range(len(self.nn.blocks)-1)], []
        NNs = [NN(self.nn.blocks[blockIdx:], self.nn.optimizer, self.nn.batch_size) for blockIdx in range(len(self.nn.blocks)-1)]
        Xs = [Function().generate_tensor(1, NNs[blockIdx].blocks[0].get_input_dim()) for blockIdx in range(len(self.nn.blocks)-1)]
        Ys = [np.zeros((Xs[blockIdx].shape[0], NNs[blockIdx].blocks[-1].get_output_dim())) for blockIdx in range(len(self.nn.blocks)-1)]
        for blockIdx in range(len(self.nn.blocks)-1):
            Ys[blockIdx][:, np.random.randint(0, Ys[blockIdx].shape[1])] = 1
        # grad_Theta = self.nn.loss_grad_Theta(X, Y, Theta)
        for _ in range(8):
            epsilon /= 2
            epsilons.append(epsilon)
            for blockIdx in range(len(self.nn.blocks) - 1):
                X = Xs[blockIdx]
                Y = Ys[blockIdx]
                curNN = NNs[blockIdx]
                f = curNN.loss(X, Y)
                # change in Theta
                d_Theta = Function().generate_tensor(*curNN.blocks[0].Theta.shape)
                d_Theta = d_Theta / np.linalg.norm(d_Theta)
                grad_Theta = curNN.grad_Theta(X, Y)
                # change type of first layer
                if isinstance(curNN.blocks[0], Residual_Layer): 
                    temp = curNN.blocks[0].Theta1
                    curNN.blocks[0].Theta1 = curNN.blocks[0].Theta1 + epsilon * d_Theta
                else:
                    temp = curNN.blocks[0].Theta
                    curNN.blocks[0].Theta = curNN.blocks[0].Theta + epsilon * d_Theta
                f_Theta_ed = curNN.loss(X, Y)
                if isinstance(curNN.blocks[0], Residual_Layer):
                    curNN.blocks[0].Theta1 = temp
                else:
                    curNN.blocks[0].Theta = temp
                diffs_Theta[blockIdx].append(np.abs(f_Theta_ed - f))
                diffs_power_Theta[blockIdx].append(np.abs(f_Theta_ed - f - epsilon * np.dot(grad_Theta.flatten(), d_Theta.flatten())))
                # change in X
                d_X = Function().generate_tensor(1, curNN.blocks[0].get_input_dim())
                d_X = d_X / np.linalg.norm(d_X)
                grad_X = curNN.grad_X(X, Y)
                f_X_ed = curNN.loss(X + epsilon * d_X, Y)
                diffs_X[blockIdx].append(np.abs(f_X_ed - f))
                diffs_power_X[blockIdx].append(np.abs(f_X_ed - f - epsilon * np.dot(grad_X.flatten(), d_X.flatten())))

        fig, axs = plt.subplots(nrows=len(self.nn.blocks)-1, ncols=2)
        fig.suptitle(f"Gradient Test, {self.nn.name} Loss")
        # plot change in Theta
        for blockIdx in range(len(self.nn.blocks)-1):
            axs[blockIdx][1].semilogy(range(8), diffs_Theta[blockIdx], linestyle='--', marker='o', label="normal")
            axs[blockIdx][1].semilogy(range(8), diffs_power_Theta[blockIdx], linestyle='--', marker='o', label="power")
            axs[blockIdx][1].legend()
            axs[blockIdx][1].set_title(f"Gradient weights & biases layer {blockIdx}")
            axs[blockIdx][1].set_xlabel('i')
        # plot change in X
        for blockIdx in range(len(self.nn.blocks)-1):
            axs[blockIdx][0].semilogy(range(8), diffs_X[blockIdx], linestyle='--', marker='o', label="normal")
            axs[blockIdx][0].semilogy(range(8), diffs_power_X[blockIdx], linestyle='--', marker='o', label="power")
            axs[blockIdx][0].legend()
            axs[blockIdx][0].set_title(f"Gradient input layer {blockIdx}")
            axs[blockIdx][0].set_xlabel('i')
        fig.show()
