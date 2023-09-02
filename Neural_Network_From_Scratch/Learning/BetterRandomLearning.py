from Models.MyNNFramework import *
import nnfs
from nnfs.datasets import vertical_data

nnfs.init()

X, y = vertical_data(100, 3)

# Creating Layers
layer1 = LayerDense(2, 3)
activation1 = ActivationReLU()
layer2 = LayerDense(layer1.size, 3)
activation2 = ActivationSoftmax()
lossFunction = CategoricalCrossEnthropy()

best_layer1_weights = layer1.weights.copy()
best_layer1_biases = layer1.biases.copy()
best_layer2_weights = layer2.weights.copy()
best_layer2_biases = layer2.biases.copy()

lowestLoss = 999999

for i in range(100000):

    layer1.weights += 0.05 * np.random.randn(2, 3)
    layer1.biases += 0.05 * np.random.randn(3)
    layer2.weights += 0.05 * np.random.randn(3, 3)
    layer2.biases += 0.05 * np.random.randn(3)

    # Forward Propagation
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    loss = lossFunction.calculate(activation2.output, y)

    if loss < lowestLoss:
        lowestLoss = loss
        print('New set of weights found, iteration:', i,
              'loss:', loss, 'acc:', lossFunction.accuracy(activation2.output, y))

        best_layer1_weights = layer1.weights.copy()
        best_layer1_biases = layer1.biases.copy()
        best_layer2_weights = layer2.weights.copy()
        best_layer2_biases = layer2.biases.copy()

    else:
        layer1.weights = best_layer1_weights.copy()
        layer1.biases = best_layer1_biases.copy()
        layer2.weights = best_layer2_weights.copy()
        layer2.biases = best_layer2_biases.copy()

print(best_layer1_weights)
print(best_layer1_biases)

print(best_layer2_weights)
print(best_layer2_biases)

