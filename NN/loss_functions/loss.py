from utils.autograd.autograd import Value
import numpy as np

def MSE(prediction, label):
    label_data = label.data if isinstance(label, Value) else label
    out = Value(0.5 * np.sum((prediction.data - label_data) ** 2), (prediction,), "mse")

    def _backward():
        prediction.grad += (prediction.data - label_data) * out.grad

    out._backward = _backward
    return out
