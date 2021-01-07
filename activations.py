import numpy as np

class ReLU:
    def __init__(self):
        pass

    def fprop(self, x):
        return np.maximum(0,x)

    def bprop(self, gradients):
        if gradients > 0:
            return 1
        else:
            return 0

class Sigmoid:
    pass


def define_activation_function(activation_function):

    available_activation_functions = ["ReLU", "Sigmoid"]
    assert activation_function in available_activation_functions, \
            f"Activation function selected is not available. Please choose from {available_activation_functions}"

    if activation_function == "ReLU":
        activation_function = ReLU()
    elif activation_function == "Sigmoid":
        activation_function = Sigmoid()

    return activation_function
