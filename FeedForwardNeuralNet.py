import numpy as np
import loss_functions
import activations
import output_layers


class FeedForwardNeuralNet:

    def __init__(self, network_shape, activation_function="ReLU", output_layer=None, loss="cross_entropy", regularisation=None):
        # network shape is a list containing the number of units per layer. This include the input and output dims.
        self.network_shape = network_shape
        self.activation_function = activations.define_activation_function(activation_function)
        self.output_layer = output_layers.define_output_layer(output_layer)
        self.loss = loss_functions.define_loss_function(loss)
        # TODO: link regularisation directly into loss function
        self.regularisation = regularisation
        self.network_dict = {}
        self._get_network_dict()


    def _get_network_dict(self):
        """
        Function obtains a dictionary that defines the neural network.
        The dict contains the weights and biases for the network and are updated during the training process.
        """

        for index in range(len(self.network_shape)-1):
            layer_number = index + 1
            matrix_input_size, matrix_output_size = self.network_shape[layer_number], self.network_shape[layer_number+1]
            self.network_dict[f"hidden_layer_{index}"] = [init_weights(matrix_input_size, matrix_output_size),
                                                     init_biases(matrix_input_size, matrix_output_size)]

        # define layers as "hidden_layer_x" : [weight and biases in here]

    def _fprop(self):
        pass


    def _forward(self, x):
        output = x
        for index in range(len(self.network_shape)):
            layer_number = index + 1
            weights, biases = self.network_dict[f"hidden_layer_{layer_number}"]
            output = self._fprop(output, weights, biases)
            if index != range(len(self.network_shape)) -1 :
                output = self.activation_function(output)

        # final layer requires different activation function depending on output
        output = self.output_layer(x)
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











