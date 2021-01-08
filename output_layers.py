# File for possible output layers and a function to obtain the correct layer given the users input



def define_output_layer(output_layer):
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
    return x

def output_sigmoid(x):
    return 1/(1+np.exp(-x))

def output_softmax():
    exp_x = np.exp(x)
    return (exp_x.T / np.sum(exp_x, axis=1)).T



