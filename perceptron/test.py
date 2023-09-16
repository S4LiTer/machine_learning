from  mnist import MNIST # 28x28
import Perceptron
import random
import numpy as np
import time


start_time = time.time()
mndata = MNIST("samples")

_images, _labels = mndata.load_training()


images = np.array(_images)/255

labels = []
for label in _labels:
    if label == 3:
        labels.append(1)
    else:
        labels.append(0)

labels = np.array(labels)
print("load:", time.time() - start_time, "s")


 ################################## END OF LOADING


perceptron = Perceptron.Perceptron(len(images[0]))
start_time = time.time()
perceptron.test_network(images[2000:12000], labels[2000:12000])

#perceptron.learn_brute_force(images[:1000], labels[:1000])
bp_weights, bp_bias = perceptron.learn_backpropagation(images[:2000], labels[:2000])

print("finish:", round(time.time() - start_time, 2), "s")

perceptron.test_network(images[2000:12000], labels[2000:12000])
