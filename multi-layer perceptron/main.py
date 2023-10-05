from  mnist import MNIST # 28x28
import matplotlib.pyplot as plt
import neural_network
import numpy as np
import time



start_time = time.time()
mndata = MNIST("samples")

_images, _labels = mndata.load_training()


images = np.array(_images)/255

labels = []
for label in _labels:
    exp = np.array([0. for _ in range(10)])
    exp[label] = 1.
    labels.append(exp)


labels = np.array(labels)
print("load:", time.time() - start_time, "s")




nn = neural_network.NeuralNetwork([784, 16, 16, 10], activation_func="relu")
# nn.storeNetwork(1, "load")
nn.Train(images, labels, 100, 1, 250)
# nn.storeNetwork(4, "save")
nn.Test(images[45000:55000], labels[45000:55000])


"""
Jednovrstva funguje skvele, vicevrstva nefunguje vubec :(
""" 