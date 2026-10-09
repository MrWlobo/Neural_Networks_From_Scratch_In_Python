from utils.autograd.autograd import Value

class Sequential:
    def __init__(self, *args):
        self.layers = []
        for layer in args:
            self.layers.append(layer)

    # Forward pass being ran locally on every layer in a sequence
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward_pass(input)

        return input
