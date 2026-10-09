from utils.autograd.autograd import Value
import numpy as np

def ReLU(x):
    x = x if isinstance(x, Value) else Value(x)

    out = Value(np.maximum(0, x.data), (x,), "relu")

    def _backward():
        x.grad += (out.data > 0) * out.grad

    out._backward = _backward
    return out
