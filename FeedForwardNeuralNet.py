import numpy as np
import loss_functions
import activations
"""
Basic idea behind the Network:
1. high level network stuff happens in this file - try to make like Keras

"""



class FeedForwardNeuralNet:

    def __init__(self, input_dim, output_dim, num_units_per_hidden_layer,
                 activation_function, loss, regularisation=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_units_per_hidden_layer = num_units_per_hidden_layer
        self.activation_function = activations.define_activation_function(activation_function)
        self.loss = loss_functions.define_loss_function(loss)
        # TODO: linke regularisation directly into loss function
        self.regularisation = regularisation
        self.weights = None
        self.bias = None


    def _forward(self, x):
        fprop, _ = self.activation_function
        output_layer = final_layer()
        for index, layer in enumerate(range(self.num_units_per_hidden_layer)):
            x = np.dot(self.weights[index].T, x) + self.bias[index]
            x = fprop(self, x)

        output = output_layer(x)
        return output



    def train(self, x, y):
        output = _forward(x)
        loss = self.loss(output, y)
        update_weights(loss)

    def test(self, x, y):
        prediction = _forward(x)
        loss = self.loss(prediction, y)


    def predict(self, x):
        prediction = _forward(x)
        return prediction


    def final_layer(self):
        pass

    def activation(self):
        if self.activation_function == "ReLU":
            fprop =
            bprop =

        elif self.activation() == "Sigmoid":
            fprop =
            bprop =
        else:
            assert "The activation function chosen is not available, please choose between ReLU or Sigmoid"

        return fprop, bprop










