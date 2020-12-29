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


    def get_network_dict(self):
        """
        Function obtains a dictionary that defines the neural network.
        The dict contains the weights and biases for the network and are updated during the training process.
        """

        # define layers as "hidden_layer_x" : [weight and biases in here]
        pass


    def _forward(self, x):
        network_dict = FeedForwardNeuralNet.get_network_dict()
        for layer in range(len(self.num_units_per_hidden_layer)):
            # do fprop in here
            # fprop, _ =
            pass

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











