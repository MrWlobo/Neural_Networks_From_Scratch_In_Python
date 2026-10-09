from utils.autograd.autograd import Value

class Sequential:
    def __init__(self, *args):
        self.layers = []
        for layer in args:
            self.layers.append(layer)

    # Forward pass being ran locally on every layer in a sequence
    def forward(self, input_data):
        input_data = input_data if isinstance(input_data, Value) else Value(input_data)
        if input_data.data.ndim == 1:
            input_data = Value(input_data.data.reshape(1, -1))
        for layer in self.layers:
            input_data = layer.forward_pass(input_data)

        return input_data

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
