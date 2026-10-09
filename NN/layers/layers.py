import numpy as np
from utils.autograd.autograd import Value

class Dense:
    def __init__(self,
                 input_count: int,
                 output_count: int,
                 activation
                 ):

        self.input_count = input_count
        self.output_count = output_count
        self.weights = Value(np.random.rand(input_count, output_count))
        self.biases = Value(np.random.rand(output_count))
        self.activation = activation
