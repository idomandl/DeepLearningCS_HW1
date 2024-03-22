import numpy as np
from function.function import Function
from network.residual_layer import Residual_Layer


class Jacobian_Transposed_Test:
    def __init__(self, func: Function):
        self.func = func

    def __call__(self, X, Theta):      
        # change in X
        u_X = self.func.generate_tensor(*X.shape)
        u_X = u_X / np.linalg.norm(u_X)
        v_X = self.func.generate_tensor(*X.shape)
        v_X = v_X / np.linalg.norm(v_X)
        grad_X = self.func.grad_X(X, Theta)
        # print (f'{grad_X.shape=}, {u_X.shape=}, {v_X.shape=}')
        return abs(u_X @ (grad_X * v_X).T - v_X @ (grad_X * u_X).T).item()


class Jacobian_Transposed_Test_Residual:
    def __init__(self, block: Residual_Layer):
        self.block = block

    def __call__(self, X, Theta1, Theta2):      
        # change in X
        u_X = Function().generate_tensor(*X.shape)
        u_X = u_X / np.linalg.norm(u_X)
        v_X = Function().generate_tensor(*X.shape)
        v_X = v_X / np.linalg.norm(v_X)
        self.Theta1 = Theta1
        self.Theta2 = Theta2
        self.block.forward(X)
        grad_X, grad_Theta1, grad_Theta2, grad1, grad2 = self.block.grad_test()
        # print (f'{grad_X.shape=}, {u_X.shape=}, {v_X.shape=}')
        return abs(u_X @ (grad_X * v_X).T - v_X @ (grad_X * u_X).T).item()
