import numpy as np
import loss_functions
import activations
import output_layers

class FeedForwardNeuralNet:

    def __init__(self, network_shape, output_layer=None, activation_function="ReLU",  loss="cross_entropy",
                 regularisation=None, batch_size=50):
        # network shape is a list containing the number of units per layer. This include the input and output dims.
        self.network_shape = network_shape
        self.output_layer = output_layers.define_output_layer(output_layer)
        self.activation_function = activations.define_activation_function(activation_function)
        self.loss = loss_functions.define_loss_function(loss)
        # TODO: link regularisation directly into loss function
        self.regularisation = regularisation
        self.batch_size = batch_size
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
            self.network_dict[f"hidden_layer_{index}"] = [self.init_weights(matrix_input_size, matrix_output_size),
                                                     self.init_biases(matrix_input_size, matrix_output_size)]

        # define layers as "hidden_layer_x" : [weight and biases in here]

    def _init_weights(self, input_size, output_size):
        # "He" initialisation - chosen because ReLU activation function is most likely to be used
        #  and "He" works best with ReLU. For more details see: https://cs231n.github.io/neural-networks-2/
        # TODO: add link to equation for He initialisation

        weights = np.random.normal(0, np.sqrt(2/input_size), size=(input_size, output_size))
        return weights

    def _init_biases(self, input_size, output_size):
        # Can intialise biases to zero. See Stanford notes for more detail: https://cs231n.github.io/neural-networks-2/

        biases = np.zeros((input_size, output_size))
        return biases


    def _fprop(self, data, weights, biases):
        output = np.dot(data, weights) + biases
        return output

    def _forward(self, x):
        output = x
        for index in range(len(self.network_shape)):
            layer_number = index + 1
            weights, biases = self.network_dict[f"hidden_layer_{layer_number}"]
            output = self._fprop(output, weights, biases)
            if index != range(len(self.network_shape)) - 1:
                output = self.activation_function(output)

        # final layer requires different activation function depending on output
        output = self.output_layer(x)
        return output



    def train(self, x, y):
        output = self._forward(x)
        loss = self.loss(output, y)
        #update_weights(loss)

    def test(self, x, y):
        prediction = self._forward(x)
        loss = self.loss(prediction, y)


    def predict(self, x):
        prediction = self._forward(x)
        return prediction











