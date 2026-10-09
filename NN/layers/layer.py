from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    def __init__(self,
                 input_count: int,
                 output_count: int,
                 activation
                 ):

        self.input_count = input_count
        self.output_count = output_count
        self.weights = np.random.rand(input_count, output_count)
        self.biases = np.random.rand(output_count)

        self.activation = activation
