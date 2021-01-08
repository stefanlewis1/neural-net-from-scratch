import numpy as np

def mean_square_error(prediction, y):
    loss = np.sqrt((prediction - y)**2)
    return loss

def cross_entropy_error(prediction, y):
    loss = -np.sum([prediction[i] * np.log2(y[i]) for i in range(len(y))])
    return loss


def define_loss_function(loss_function_string):

    available_loss_functions = ["mse", "cross_entropy"]
    assert loss_function_string in available_loss_functions,\
        f"Activation function selected is not available. Please choose from {available_loss_functions}"

    if loss_function_string == "mse":
        loss_function = mean_square_error
    elif loss_function_string == "cross_entropy":
        loss_function = cross_entropy_error

    return loss_function


