import numpy as np
from utils.autograd.autograd import Value

class Dense:
    def __init__(self,
                 input_count: int,
                 output_count: int,
                 activation=None
                 ):

        self.input_count = input_count
        self.output_count = output_count
        self.weights = Value(np.random.rand(input_count, output_count))
        self.biases = Value(np.random.rand(1, output_count))
        self.activation = activation

    def forward_pass(self, input):
        output = input.matmul(self.weights) + self.biases
        if self.activation is not None:
            output = self.activation(output)
        return output
