# File for possible output layers and a function to obtain the correct layer given the users input
import numpy as np


def define_output_layer(output_layer):
    f"""
    Given string input the function returns the appropiate output layer to be used in the neural network model.

    :param output_layer: string input given by user to define the output layer of the neural network.
    User can select from {available_output_layers}
    :type output_layer: str

    :return: function that is used to determines the final layer of the network.
    :rtype: function
    """
    available_output_layers = ["linear", "sigmoid", "softmax"]
    assert output_layer in available_output_layers, \
        f"Output layer selected is not available. Please choose from {available_output_layers}"

    if output_layer == "linear":
        output_layer = output_linear
    elif output_layer == "sigmoid":
        output_layer = output_sigmoid
    elif output_layer == "softmax":
        output_layer = output_softmax

    return output_layer

def output_linear(x):
    """
    Function returns its input to generate a linear layer.

    :param x: input to the final layer.
    :type x: ndarray

    :return: the same as its input
    :rtype: ndarray
    """
    return x

def output_sigmoid(x):
    """
    Function to pass the final layer through a sigmoid function element-wise. Used for binary classification.

    :param x: input to the final layer
    :type x: ndarray

    :return: input passed through sigmoid function element-wise
    :rtype: ndarray
    """
    return 1/(1+np.exp(-x))

def output_softmax(x):
    """
    Function to pass the final layer through a softmax function element-wise. Used for multiclass classification.

    :param x: input to the final layer
    :type: ndarray

    :return: input passed through softmax function element-wise
    :rtype: ndarray
    """
    exp_x = np.exp(x)
    return (exp_x.T / np.sum(exp_x, axis=1)).T



