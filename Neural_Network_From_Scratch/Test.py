import numpy as np

inputs = np.array([[1, 2, 3, 2.5],
 [2., 5., -1., 2],
 [-1.5, 2.7, 3.3, -0.8]])

weights = np.array([[0.2, 0.8, -0.5, 1],
 [0.5, -0.91, 0.26, -0.5],
 [-0.26, -0.27, 0.17, 0.87]]).T

biases = np.array([[2, 3, 0.5]])

#forward pass
layer_outputs = np.dot(inputs, weights) + biases
layer_activation = np.maximum(0, layer_outputs)
dvalues = layer_activation

# Backward pass - for this example we're using ReLU's output
# as passed-in gradients (we're minimizing this output)
dReLU = dvalues.copy()
dReLU[layer_outputs < 0] = 0

dinputs = np.dot(dReLU, weights.T)
dweights = np.dot(inputs.T, dReLU)
dbiases = np.sum(dReLU, axis=0, keepdims=True)

weights += -0.001 * dweights
biases += -0.001 * dbiases



print(weights)
print(biases)

