# File for possible output layers and a function to obtain the correct layer given the users input



def define_output_layer(output_layer):
    available_output_layers = ["Linear", "Sigmoid", "Softmax"]
    assert output_layer in available_output_layers, \
        f"Output layer selected is not available. Please choose from {available_output_layers}"

    if output_layer == "Linear":
        output_layer = output_linear
    elif output_layer == "Sigmoid":
        output_layer = output_sigmoid
    elif output_layer == "Softmax":
        output_layer = output_softmax

    return output_layer

def output_linear(x):
    return x

def output_sigmoid():
    pass

def output_softmax():
    pass



