This repo writes a feed forward neural network from scratch which is packaged into a simple library.

The final implementation should be easy to use as sklearn, an example is shown below:

import NeuralNet \n
train_x, train_y, val_x, val_y = split_data(data) \n

model = NeuralNet.FeedForwardNeuralNet() \n
model.train(train_x,train_y) \n
model.test(val_x,val_y) \n
model.predict(x) \n

The code is written in a modular way that will allow the neural network to be expanded as required.


