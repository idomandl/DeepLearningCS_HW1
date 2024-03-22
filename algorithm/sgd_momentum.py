from algorithm.sgd import SGD
class SGDMomentum(SGD):

    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.velocities = {}

    def update_params(self, Theta_grads, layer_id ,*args):
        if layer_id not in self.velocities:
            self.velocities[layer_id] = 0
        self.velocities[layer_id] = self.momentum * self.velocities[layer_id] + (1-self.momentum)*Theta_grads
        return - self.velocities[layer_id]*self.learning_rate

