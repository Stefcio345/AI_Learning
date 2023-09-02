import numpy as np


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


class LayerDense:
    size = None

    def __init__(self, input_size, n_neurons, l1_weight_lambda = 0, l1_bias_lambda=0, l2_weight_lambda=0, l2_bias_lambda=0):
        self.size = n_neurons
        self.weights = np.random.randn(input_size, n_neurons) * 0.01
        self.biases = np.zeros(n_neurons)

        #regularization (to prevent weights and biases from skyrocketing)
        self.weight_lam_l1 = l1_weight_lambda
        self.bias_lam_l1 = l1_bias_lambda
        self.weight_lam_l2 = l2_weight_lambda
        self.bias_lam_l2 = l2_bias_lambda


    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        #gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)[0]

        #Gradient regularization
        #L1 regularization (linear)
        if self.weight_lam_l1 > 0:
            dL1 = np.one_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_lam_l1 * dL1
        if self.bias_lam_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_lam_l2 * dL1

        #L2 regularization (squared)
        if self.weight_lam_l2 > 0:
            self.dweights += 2 * self.weight_lam_l2 * self.weights
        if self.bias_lam_l2 > 0:
            self.dbiases += 2 * self.bias_lam_l2 * self.biases


        #Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    def print(self):
        print("--------------Layer--------------")
        print("Weights: ", self.weights)
        print("Biases: ", self.biases)

class LayerDropout:

    def __init__(self, dropout_rate):
        self.rate = 1 - dropout_rate

    def forward(self, inputs):
        self.inputs = inputs
        #Generate binomial binary mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        #Applay mask to output
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

class ActivationReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dReLU = dvalues.copy()
        self.dReLU[self.inputs <= 0] = 0


class ActivationSoftmax:
    def forward(self, inputs):
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = expValues / np.sum(expValues, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    def calculate(self, outputs, target):
        sampleLosses = self.forward(outputs, target)
        dataLoss = np.mean(sampleLosses)
        return dataLoss

    def accuracy(outputs, target):
        predictions = np.argmax(outputs, axis=1)
        accuracy = np.mean(predictions == target)
        return accuracy

    def regularization_loss(self, layer):

        regularizationLoss = 0

        if layer.weight_lam_l1 > 0:
            regularizationLoss += layer.weight_lam_l1 * np.sum(np.abs(layer.weights))

        if layer.bias_lam_l1 > 0:
            regularizationLoss += layer.bias_lam_l1 * np.sum(np.abs(layer.biases))

        if layer.weight_lam_l2 > 0:
            regularizationLoss += layer.weight_lam_l2 * np.sum(layer.weights * layer.weights)

        if layer.bias_lam_l2 > 0:
            regularizationLoss += layer.bias_lam_l2 * np.sum(layer.biases * layer.biases)

        return regularizationLoss

class CategoricalCrossEnthropy(Loss):
    def forward(self, prediction, target):
        samples = len(prediction)
        prediciton_clipped = np.clip(prediction, 1e-7, 1 - 1e-7)

        if len(target.shape) == 1:
            correct_confidence = prediciton_clipped[range(samples), target]
        elif len(target.shape) == 2:
            correct_confidence = np.sum(prediciton_clipped * target, axis=1)

        likelihood = -np.log(correct_confidence)
        return likelihood

    def backward(self, dvalue, target):
        samples = len(dvalue)
        labels = len(dvalue[0])

        if len(target.shape) == 1:
            target = np.eye(labels)[target]

        self.dinputs = -target / dvalue

        self.dinputs = self.dinputs / samples


class ActivationSoftmaxLossCategoricalCrossEntropy:
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = CategoricalCrossEnthropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class OptimizerSGD:

    def __init__(self, learningRate=1., decay=0., momentum=0.):
        self.learningRate = learningRate
        self.current_learning_rate = learningRate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def preUpdate(self):
        if self.decay:
            self.current_learning_rate = self.learningRate * (1. / (1. + self.decay * self.iterations))

    def updateParams(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weightMomentums'):
                layer.weightMomentums = np.zeros_like(layer.weights)
                layer.biasMomentums = np.zeros_like(layer.biases)

            weightUpdates = self.momentum * layer.weightMomentums - self.learningRate * layer.dweights
            layer.weightMomentums = weightUpdates

            biasUpdates = self.momentum * layer.biasMomentums - self.learningRate * layer.dbiases
            layer.biasMomentums = biasUpdates

        else:
            weightUpdates = -self.current_learning_rate * layer.dweights
            biasUpdates = -self.current_learning_rate * layer.dbiases

        layer.weights += weightUpdates
        layer.biases += biasUpdates

    def postUpdate(self):
        self.iterations += 1


class AdaGradOptimizer:

    def __init__(self, learningRate=1., decay=0., epsilon=1e-7):
        self.learningRate = learningRate
        self.current_learning_rate = learningRate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def preUpdate(self):
        if self.decay:
            self.current_learning_rate = self.learningRate * (1. / (1. + self.decay * self.iterations))

    def updateParams(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    def postUpdate(self):
        self.iterations += 1


class RMSPropOptimizer:

    def __init__(self, learningRate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learningRate = learningRate
        self.current_learning_rate = learningRate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def preUpdate(self):
        if self.decay:
            self.current_learning_rate = self.learningRate * (1. / (1. + self.decay * self.iterations))

    def updateParams(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    def postUpdate(self):
        self.iterations += 1


class AdamOptimizer:

    def __init__(self, learningRate=0.001, decay=0., epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.learningRate = learningRate
        self.current_learning_rate = learningRate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def preUpdate(self):
        if self.decay:
            self.current_learning_rate = self.learningRate * (1. / (1. + self.decay * self.iterations))

    def updateParams(self, layer):

        if not hasattr(layer, 'weightCache'):
            layer.weightMomentums = np.zeros_like(layer.weights)
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasMomentums = np.zeros_like(layer.biases)
            layer.biasCache = np.zeros_like(layer.biases)

        layer.weightMomentums = self.beta1 * \
                                 layer.weightMomentums + \
                                 (1 - self.beta1) * layer.dweights
        layer.biasMomentums = self.beta1 * \
                               layer.biasMomentums + \
                               (1 - self.beta1) * layer.dbiases

        weightMomentumsCorrected = layer.weightMomentums / (1 - self.beta1 ** (self.iterations + 1))
        biasMomentumsCorrected = layer.biasMomentums / (1 - self.beta1 ** (self.iterations + 1))

        layer.weightCache = self.beta2 * layer.weightCache + (1 - self.beta2) * layer.dweights**2
        layer.biasCache = self.beta2 * layer.biasCache + (1 - self.beta2) * layer.dbiases**2

        weightCacheCorrected = layer.weightCache / (1 - self.beta2 ** (self.iterations + 1))
        biasCacheCorrected = layer.biasCache / (1 - self.beta2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * \
                         weightMomentumsCorrected / \
                         (np.sqrt(weightCacheCorrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        biasMomentumsCorrected / \
                        (np.sqrt(biasCacheCorrected) + self.epsilon)

    def postUpdate(self):
        self.iterations += 1

