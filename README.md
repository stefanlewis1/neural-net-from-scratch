Below is the main idea for top level code for this library, it should be as easy to implement as sklearn:

"""
import NeuralNet
train_x, train_y, val_x, val_y = split_data(data)

model = NeuralNet.FeedForwardNeuralNet()
model.train(train_x,train_y)
model.test(val_x,val_y)
model.predict(x)
"""