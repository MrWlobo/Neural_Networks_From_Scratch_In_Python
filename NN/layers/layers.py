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

    def forward_pass(self, input_data):
        input_data = input_data if isinstance(input_data, Value) else Value(input_data)

        output = input_data.matmul(self.weights) + self.biases
        if self.activation is not None:
            output = self.activation(output)
        return output

    def parameters(self):
        return [self.weights, self.biases]


class Flatten:
    def forward_pass(self, input_data):
        input_data = input_data if isinstance(input_data, Value) else Value(input_data)

        original_shape = input_data.data.shape
        # Flatten everything except the batch dimension
        flattened_data = input_data.data.reshape(original_shape[0], -1)

        out = Value(flattened_data, (input_data,), "flatten")

        def _backward():
            # Reshape the incoming gradients back to the original image dimensions
            input_data.grad += out.grad.reshape(original_shape)

        out._backward = _backward
        return out

    def parameters(self):
        return []
