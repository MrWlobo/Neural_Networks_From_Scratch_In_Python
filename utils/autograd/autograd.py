import numpy as np


class Value:
    """
    A single node in a NumPy-based computational graph, supporting dynamic autograd.

    Attributes:
        data (np.ndarray): The actual numerical values, which can represent raw inputs,
            learnable parameters (weights/biases), intermediate activations, or the final loss.

        grad (np.ndarray): The accumulated gradient of the loss with respect to this node,
            matched in shape to `data` and used later by an optimizer.

        _backward (callable): A closure function that executes the chain rule, accumulating
            gradients via '+=' to support branching paths in the computational graph.

        _prev (set): A set of parent `Value` instances upon which this current node was built
            (empty for primitive input/weight nodes).

        _op (str): A string representing the operation sign that created this node (e.g., '+', '*'),
            primarily used for debugging and visualization.
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # Handling basic operators: +, * and **
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        """
        The whole chain of the Chain Rule has already been stored in out.grad, except
        for the last link. However, the derivative of the addition of 2 values with
        respect to one of them will yield 1 (Exception: addition of `x + x`, in which
        case the cumulative gradient will correctly assign the value 2)
        """
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        """
        When deriving a `x * y` multiplication with respect to one of the variables
        the result will be another one. After that, multiplying by the value of the
        rest of the chain will yield the correct gradient.
        """
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        out = Value(self.data ** other, (self,), f"**{other}")

        """
        Like in previous examples, the local derivative of the function is
        calculated and its results are multiplied by the value of the
        rest of the chain.
        """
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    # Handling matrix multiplication for neural networks
    def matmul(self, other):
        out = Value(np.dot(self.data, other.data), (self, other), "matmul")

        def _backward():
            self.grad += np.dot(out.grad, other.data.T)
            other.grad += np.dot(self.data.T, out.grad)

        out._backward = _backward
        return out

    # Handling reverse order of values in operations
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"Data:\n{self.data}\n\nGradient:\n{self.grad}"

    def backward(self):
        topology = []
        visited = set()

        # Recursively builds a topology of a graph starting from the last node (Loss function value)
        def build_topology(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topology(child)
                topology.append(node)

        build_topology(self)

        # Setting Loss gradient to 1 (dL/dL = 1), then iteratively calculating gradients going toward input layers
        self.grad = np.ones_like(self.data)
        for node in reversed(topology):
            node._backward()
