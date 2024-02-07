from  mnist import MNIST # 28x28
from utils import preprocessing
import matplotlib.pyplot as plt
import neural_network
import numpy as np
import layers
import time



start_time = time.time()
mndata = MNIST("samples")

_testing_images, _testing_labels = mndata.load_testing()
testing_images = np.array(_testing_images)/255

testing_images = testing_images.reshape((testing_images.shape[0], ) + (1, 28, 28))

testing_labels = []
for label in _testing_labels:
    exp = np.array([0. for _ in range(10)])
    exp[label] = 1.
    testing_labels.append(exp)
testing_labels = np.array(testing_labels)


_images, _labels = mndata.load_training()
images = np.array(_images)/255
images = images.reshape((images.shape[0], ) + (1, 28, 28))


labels = []
for label in _labels:
    exp = np.array([0. for _ in range(10)])
    exp[label] = 1.
    labels.append(exp)
labels = np.array(labels)
print("load:", time.time() - start_time, "s")




nn = neural_network.NeuralNetwork((1, 28, 28), False)
nn.add_convolutional_layer((3, 3), 1)
nn.add_pooling_layer((2, 2))
nn.add_flattening_layer()
nn.add_fully_connected_layer(10, "sigmoid")

nn.Train(images[:3000], labels[:3000], testing_images, testing_labels, 100, 0.004, 8)

print("\nTraining samples:")
nn.Test(images[5000:6000], labels[5000:6000])

print("\nTesting samples:")
nn.Test(testing_images[:1000], testing_labels[:1000])
