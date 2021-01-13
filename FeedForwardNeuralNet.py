import numpy as np
import loss_functions
import activations
import output_layers

class FeedForwardNeuralNet:
    """ Feed forward neural network object which can train or predict in one line of code """

    def __init__(self, network_shape, output_layer="linear", activation_function="relu",  loss="sum_of_squares",
                 regularisation=0):
        """Create a new neural network instance.

        :param network_shape: the shape of the network defined by a list.
        Each entry in the list is a layer while its value is the number of units in said layer.
        Note - network_shape must include input and output layers/dimensions
        :type network_shape: list

        :param output_layer: string input to determine the final layer type. Can currently choose from linear, sigmoid
        or softmax
        :type output_layer: str

        :param activation_function: string input to determine the activation function that will be used on all but the
        output layer. Can currently choose from relu or sigmoid.
        :type activation_function: str

        :param loss: string input to determine which loss function will be used. Can currently choose from
        sum_of_squares or cross_entropy.
        :type loss: str

        :param regularisation: float input to determine lambda parameter for the L2 regularisation. Default is 0.
        :type regularisation: float

        Example:

            model = FeedForwardNeuralNetwork(network_shape=[784,100,100,10])
            model.train(x, y, epochs=10)
            model.predict(new_data_point)

        """
        # network shape is a list containing the number of units per layer. This include the input and output dims.
        self.network_shape = network_shape
        self.output_layer = output_layers.define_output_layer(output_layer)
        self.activation_function = activations.define_activation_function(activation_function)
        self.loss = loss_functions.define_loss_function(loss)
        # TODO: link regularisation directly into loss function
        self.regularisation = regularisation
        self.network_dict = self._get_network_dict()



    def _init_weights(self, input_size, output_size):
        """
        Function to initialise weights of the network using 'He' intialisation. 'He' method chosen due to effectiveness
        with ReLU activation function. For more information see https://cs231n.github.io/neural-networks-2/.

        :param input_size: input size of the layer
        :type input_size: int

        :param output_size: output size of the later
        :type output_size: int

        :return: weights for a given layer
        :rtype: ndarray

        """
        # "He" initialisation chosen because ReLU activation function works best with He initialisation.
        # ReLU is most likely activation function to be used.
        # For more details on "He" init see: https://cs231n.github.io/neural-networks-2/

        weights = np.random.normal(0, np.sqrt(2/input_size), size=(input_size, output_size))
        return weights

    def _init_biases(self, input_size, output_size):
        """
        Function to intialise the biases of the network. All intialised to zero, for more information see:
        https://cs231n.github.io/neural-networks-2//

        :param input_size: input size of layer
        :type input_size: int

        :param output_size: output size of layer
        :type output_size: int

        :return: biases for a given layer
        :rtype: ndarray
        """

        biases = np.zeros((input_size, output_size))
        return biases


    def _get_network_dict(self):
        """
        Function that obtains a dictionary that defines the neural network.
        The dict contains the weights and biases for the network which are updated during the training process.


        :return: dictionary containing parameters of the model
        :rtype: dict
        """


        network_dict = {}
        input_dimensionality = self.network_shape[0]
        for index in range(len(self.network_shape)-1):
            layer_number = index + 1
            matrix_input_size, matrix_output_size = self.network_shape[index], self.network_shape[index+1]
            network_dict[f"layer_{layer_number}"] = [self._init_weights(matrix_input_size, matrix_output_size),
                                                     self._init_biases(input_dimensionality, matrix_output_size)]

        return network_dict


    def _fprop(self, data, weights, biases):
        """
        Function to provide a single affine transformation to the data input using network parameters.

        :param data: input data that will be pass forward through the transformation
        :type data: ndarray

        :param weights: weights for the given layer
        :type weights: ndarray

        :param biases: biases for the given layer
        :type biases: ndarray

        :return: output of the data after affine transformation
        :rtype: ndarray
        """
        output = np.dot(data, weights) + biases
        return output

    def _forward(self, x):
        """
        Forward pass through the neural network. Calling this function will take the data - x - and forward propagate
        it through the network.

        :param x: data input that is forward propagated through the network.
        :type x: ndarray

        :return: network prediction using the data input
        :rtype: ndarray
        """

        output = x
        for index in range(len(self.network_shape)-1):
            layer_number = index + 1
            weights, biases = self.network_dict[f"layer_{layer_number}"]
            output = self._fprop(output, weights, biases)
            # don't want to apply activation function on final layer, using if statement to check if this is last layer
            if index != len(self.network_shape) - 2:
                output = self.activation_function.fprop(output)

        # final layer requires different activation function depending on output
        output = self.output_layer(output)
        return output



    def train(self, data, targets, epochs=100, batch_size=1):
        """
        Train the neural network using the data input and targets by calling this function.

        :param data: input data to the model
        :type data: ndarray

        :param targets: corresponding labels to the data input
        :type targets: ndarray

        :param epochs: number of times the data will be iterated over during training. Default is 100 epochs
        :type epochs: int

        :param batch_size: the number of data points used in one iteration of training. Default batch size is 1
        :type batch_size: int

        :return: None
        """
        output = self._forward(data)
        # TODO: need to ensure output layer is sigmoid/softmax when using cross_entropy loss
        loss = self.loss.error(output, targets)
        #update_weights(loss)
        return output

    def test(self, x, y):
        prediction = self._forward(x)
        loss = self.loss.error(prediction, y)


    def predict(self, x):
        prediction = self._forward(x)
        return prediction


# little test to see if the fprop is working
if __name__ == "__main__":
    model = FeedForwardNeuralNet(network_shape=[2,5,20,1])
    out = model.train([[1,1],[2,2]],[[1],[2]])
    print(out)











