import matplotlib.pyplot as plt
import numpy as np
from Models import MyNNFramework
from nnfs.datasets import spiral_data
import nnfs

from Models.MyNNFramework import *

nnfs.init()


X, y = spiral_data(100, 3)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()

# Creating Layers
layer1 = LayerDense(2, 3)
activation1 = ActivationReLU()

layer2 = LayerDense(layer1.size, 3)
activation2 = ActivationSoftmax()

lowestLoss = 999999

#Training - winiary
for i in range(int(eval("1e6"))):

    layer1.weights = np.random.randn(2, 3)*10
    layer1.biases = np.random.randn(1, 3)*10
    layer2.weights = np.random.randn(3, 3)*3
    layer2.biases = np.random.randn(1, 3)*3

    # Forward Propagation
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    # Loss Calculation
    lossFuncton = CategoricalCrossEnthropy()
    loss = lossFuncton.calculate(activation2.output, y)


    if loss < lowestLoss:
        lowestLoss = loss
        print('New set of weights found, iteration:', i,
            'loss:', loss, 'acc:', str(round(lossFuncton.accuracy(activation2.output, y)*100, 2)) + "%")

        best_layer1_weights = layer1.weights.copy()
        best_layer1_biases = layer1.biases.copy()
        best_layer2_weights = layer2.weights.copy()
        best_layer2_biases = layer2.biases.copy()

print(best_layer1_weights)
print(best_layer1_biases)

print(best_layer2_weights)
print(best_layer2_biases)