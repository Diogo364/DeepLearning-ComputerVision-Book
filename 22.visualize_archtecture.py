from nn.conv.lenet import LeNet
from tensorflow.keras.utils import plot_model


# initialize the LeNet and then write the network achitecture
# visualization graph to disk
model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file='./outputs/22.lenet.png', show_shapes=True)