import numpy as np


class mean_square:

    def mean_square_error(self, prediction, y):
        loss = ((prediction - y)**2) / len(prediction)
        return loss

    # TODO: need to work out gradients and backprop

class cross_entropy:

    def cross_entropy_error(self, prediction, y):
        return -np.sum(y * np.log2(prediction))

    # TODO: need to work out gradients and backprop



def define_loss_function(loss_function_string):

    available_loss_functions = ["mse", "cross_entropy"]
    assert loss_function_string in available_loss_functions,\
        f"Activation function selected is not available. Please choose from {available_loss_functions}"

    if loss_function_string == "mse":
        loss_function = mean_square
    elif loss_function_string == "cross_entropy":
        loss_function = cross_entropy

    return loss_function


