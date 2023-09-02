from Models.MyNNFramework import *
import nnfs

X, y = spiral_data(1000, 3)

layer1 = LayerDense(2, 64, l2_weight_lambda=5e-4, l2_bias_lambda=5e-4)
activation1 = ActivationReLU()
layer2_dropout = LayerDropout(0.1)
layer3 = LayerDense(layer1.size, 3)

loss_activation = ActivationSoftmaxLossCategoricalCrossEntropy()

# optimizer
optimizer = AdamOptimizer(learningRate=0.05, decay=5e-5)

for epoch in range(10001):
    # Forward pass
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2_dropout.forward(activation1.output)
    layer3.forward(layer2_dropout.output)

    loss_data  = loss_activation.forward(layer3.output, y)
    loss_regular = loss_activation.loss.regularization_loss(layer1) + loss_activation.loss.regularization_loss(layer3)

    loss = loss_data + loss_regular

    acc = CategoricalCrossEnthropy.accuracy(loss_activation.output, y)

    if not epoch % 100:
        print("Epoch: ", epoch, "Loss: ", round(loss, ndigits=4), "Acc: ", round(acc, ndigits=3), "Learning rate: ", round(optimizer.current_learning_rate, ndigits=6), "regular loss: ", round(loss_regular, ndigits=3), "data loss: ", round(loss_data, ndigits=3))

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    layer3.backward(loss_activation.dinputs)
    layer2_dropout.backward(layer3.dinputs)
    activation1.backward(layer2_dropout.dinputs)
    layer1.backward(activation1.dReLU)

    optimizer.preUpdate()
    optimizer.updateParams(layer1)
    optimizer.updateParams(layer3)
    optimizer.postUpdate()

#Test phase on out of sample data

X_test, y_test = spiral_data(100, 3)
layer1.forward(X_test)
activation1.forward(layer1.output)
layer2_dropout.forward(activation1.output)
layer3.forward(layer2_dropout.output)
loss = loss_activation.forward(layer3.output, y_test)
acc = CategoricalCrossEnthropy.accuracy(loss_activation.output, y_test)

print(f"validation, acc: {acc:.3f}, loss: {loss:.3f}")
