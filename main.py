from NN.layers.layers import Dense
from NN.activations.activation import ReLU
from NN.networks.networks import Sequential

dense1 = Dense(3, 5, ReLU)
dense2 = Dense(5, 7, ReLU)
dense3 = Dense(7, 1, ReLU)

model = Sequential(dense1, dense2, dense3)
print(model.forward([1, 2, 3]))
