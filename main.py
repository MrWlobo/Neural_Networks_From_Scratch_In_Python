from NN.layers.layers import Dense
from NN.activations.activation import ReLU

dense = Dense(3, 5, ReLU)

print(dense.weights)