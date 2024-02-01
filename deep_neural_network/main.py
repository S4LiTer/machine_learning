from  mnist import MNIST # 28x28
import matplotlib.pyplot as plt
import preprocessing
import neural_network
import numpy as np
import time



start_time = time.time()
mndata = MNIST("samples")

_testing_images, _testing_labels = mndata.load_testing()
testing_images = np.array(_testing_images)/255

testing_labels = []
for label in _testing_labels:
    exp = np.array([0. for _ in range(10)])
    exp[label] = 1.
    testing_labels.append(exp)
testing_labels = np.array(testing_labels)


_images, _labels = mndata.load_training()
images = np.array(_images)/255

labels = []
for label in _labels:
    exp = np.array([0. for _ in range(10)])
    exp[label] = 1.
    labels.append(exp)
labels = np.array(labels)
print("load:", time.time() - start_time, "s")




nn = neural_network.NeuralNetwork(784)
nn.add_layer(128, "relu")
nn.add_layer(64, "relu")
nn.add_layer(10, "sigmoid")
print("Network structure:", nn.layer_sizes)

images = preprocessing.preprocess_array(images)
testing_images = preprocessing.preprocess_array(testing_images)
nn.Train(images, labels, testing_images, testing_labels, 100, 0.002, 10)
#nn.storeNetwork(1, "save")

print("\nTraining samples:")
nn.Test(images[:10000], labels[:10000])

print("\nTesting samples:")
nn.Test(testing_images, testing_labels)
