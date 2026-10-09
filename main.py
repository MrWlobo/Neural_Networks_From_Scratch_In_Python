from NN.layers.layers import Dense
from NN.activations.activation import ReLU
from NN.networks.networks import Sequential
from NN.loss_functions.loss import MSE

dense1 = Dense(3, 5, ReLU)
dense2 = Dense(5, 7, ReLU)
dense3 = Dense(7, 1, ReLU)

label = 10
model = Sequential(dense1, dense2, dense3)

prediction = model.forward([1, 2, 3])
loss = MSE(prediction, label)
loss.backward()
print(loss)
