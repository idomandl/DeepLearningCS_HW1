import numpy as np
from softmax_loss import SoftmaxLoss, SoftmaxLossLoop
X = np.array([[2.6, -1.7, 1], [-0.6, -1.7, 1], [-0.06, 1.59, 1], [1.93, -0.64417, 1]])
THETA = np.array([[0.8,0.22,0.38,-0.08,-1.599],[0.004,1.38,-0.16,-0.47,-0.33],[-0.05, 0.05, 0.446,-1.4,-0.47]])
Y = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 1], [0,0,0,0], [0,0,1,0]]).T
softmax_loss = SoftmaxLoss(THETA, Y)
print(softmax_loss(X))
print(softmax_loss.calc_grad(X))