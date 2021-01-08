import numpy as np

class ReLU:

    def fprop(self, x):
        return np.maximum(0,x)

    def bprop(self, gradients):
        if gradients > 0:
            return 1
        else:
            return 0

class Sigmoid:

    def fprop(self, x):
        return 1 / (1 + np.exp(-x))

    def bprop(self, gradients):
        return self.fprop(gradients) * (1 - self.fprop(gradients))



def define_activation_function(activation_function):

    available_activation_functions = ["relu", "sigmoid"]
    assert activation_function in available_activation_functions, \
            f"Activation function selected is not available. Please choose from {available_activation_functions}"

    if activation_function == "relu":
        activation_function = ReLU()
    elif activation_function == "sigmoid":
        activation_function = Sigmoid()

    return activation_function
