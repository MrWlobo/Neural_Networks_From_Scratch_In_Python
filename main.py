from NN.layers.layers import Dense
from NN.activations.activation import ReLU
from NN.networks.networks import Sequential
from NN.loss_functions.loss import MSE
from NN.optimizers.optimizers import SGD

dense1 = Dense(3, 5, ReLU)
dense2 = Dense(5, 7, ReLU)
dense3 = Dense(7, 1)

label = 10
model = Sequential(dense1, dense2, dense3)
optimizer = SGD(model.parameters(), learning_rate=0.02)
epochs = 100

for epoch in range(epochs):
    prediction = model.forward([1, 2, 3])
    loss = MSE(prediction, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1} MSE Loss value: {loss.data}")
