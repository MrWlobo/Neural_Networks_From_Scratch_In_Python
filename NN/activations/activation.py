from utils.autograd.autograd import Value
import numpy as np

def ReLU(input):
    out = Value(np.maximum(0, input.data), (input,), "relu")

    def _backward():
        input.grad += (out.data > 0) * out.grad

    out._backward = _backward
    return out
