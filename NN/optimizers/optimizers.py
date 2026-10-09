import numpy as np

class SGD:
    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            param.data -= (param.grad * self.learning_rate)

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.data)
