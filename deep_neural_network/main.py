from  mnist import MNIST # 28x28
from utils import preprocessing
import matplotlib.pyplot as plt
import neural_network
import numpy as np
import emnist
import layers
import time



images, raw_labels = emnist.extract_training_samples("balanced")
testing_images, raw_testing_labels = emnist.extract_test_samples("balanced")


images = images.reshape((-1, 1, 28, 28)) / 255
labels = np.zeros((raw_labels.shape[0], np.max(raw_labels)+1))
labels[np.arange(len(raw_labels)), raw_labels] = 1


testing_images = testing_images.reshape((-1, 1, 28, 28)) / 255
testing_labels = np.zeros((raw_testing_labels.shape[0], np.max(raw_testing_labels)+1))
testing_labels[np.arange(len(raw_testing_labels)), raw_testing_labels] = 1


nn = neural_network.NeuralNetwork((1, 28, 28), True)
nn.storeNetwork(17, "load")


nn.add_convolutional_layer((3, 3), 8)
nn.add_pooling_layer((2, 2))
nn.add_convolutional_layer((2, 2), 16)
nn.add_pooling_layer((2, 2))
nn.add_flattening_layer()
nn.add_fully_connected_layer(256, "relu")
nn.add_fully_connected_layer(128, "relu")
nn.add_fully_connected_layer(47, "softmax")


nn.Train(images, labels, testing_images, testing_labels, 1, 0.0002, 3)
nn.storeNetwork(17)

charmap_path = "samples/EMNIST/emnist-balanced-mapping.txt"
neural_network.Test(nn, testing_images, testing_labels, charmap_path)


