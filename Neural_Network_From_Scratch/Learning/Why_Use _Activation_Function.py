import math

import matplotlib.pyplot as plt
import numpy as np
import nnfs

#nnfs.init()


def spiral_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


class Layer:
    size = None

    def __init__(self, input_size, n_neurons):
        self.size = n_neurons
        self.weights = np.random.randn(input_size, n_neurons) * 0.1
        self.biases = np.random.randn(n_neurons)*0.1
        print(self.biases)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def print(self):
        print("--------------Layer--------------")
        print("Weights: ", self.weights)
        print("Biases: ", self.biases)


class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class ActivationSoftmax:
    def forward(self, inputs):
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = expValues / np.sum(expValues, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, outputs, target):
        sampleLosses = self.forward(outputs, target)
        dataLoss = np.mean(sampleLosses)
        return dataLoss

    def accuracy(outputs, target):
        predictions = np.argmax(outputs ,axis=1)
        accuracy = np.mean(predictions == target)
        return accuracy



class CategoricalCrossEnthropy(Loss):
    def forward(self, prediction, target):
        samples = len(prediction)
        prediciton_clipped = np.clip(prediction, 1e-7, 1-1e-7)

        if len(target.shape) == 1:
            correct_confidence = prediciton_clipped[range(samples), target]
        else:
            correct_confidence = np.sum(prediciton_clipped*target, axis=1)

        likelihood = -np.log(correct_confidence)
        return likelihood


X, y = spiral_data(200, 1)

X = X[:, 0]

X.shape = (200,1)

# Creating Layers
layer1 = Layer(1, 5)
activation1 = ActivationReLU()

layer2 = Layer(layer1.size, 5)
activation2 = ActivationReLU()

OutputLayer = Layer(layer2.size, 1)
activationOutput = ActivationReLU()

# Forward Propagation
layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

OutputLayer.forward(activation2.output)
activationOutput.forward(OutputLayer.output)

# Loss Calculation
#lossFuncton = CategoricalCrossEnthropy()
#loss = lossFuncton.calculate(activation2.output, y)

#print(loss)
#print(lossFuncton.accuracy(activation2.output, y))

plt.scatter(X, activationOutput.output)
plt.show()
