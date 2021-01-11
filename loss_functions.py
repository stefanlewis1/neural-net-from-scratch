import numpy as np


class SumOfSquares:

    def error(self, prediction, y):
        # TODO: Add ref to this and cross_entropy
        return np.mean(np.sum((prediction - y)**2, axis=1)) * 0.5

    # TODO: need to work out gradients and backprop
    def gradients(self, prediction, y):

        return (prediction - y) / prediction.shape[0]

class CrossEntropy:

    def error(self, prediction, y):
        return -np.mean(np.sum(y * np.log2(prediction)), axis=1)

    # TODO: need to work out gradients and backprop
    def gradients(self, prediction, y):
        return -(y / prediction) / prediction.shape[0]


def define_loss_function(loss_function_string):

    available_loss_functions = ["sum_of_squares", "cross_entropy"]
    assert loss_function_string in available_loss_functions,\
        f"Activation function selected is not available. Please choose from {available_loss_functions}"

    if loss_function_string == "sum_of_squares":
        loss_function = SumOfSquares()
    elif loss_function_string == "cross_entropy":
        loss_function = CrossEntropy()

    return loss_function


