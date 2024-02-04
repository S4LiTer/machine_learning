from utils import activation_functions as act_funcs
import matplotlib.pyplot as plt
import neural_network
from mnist import MNIST
import numpy as np
import layers
import json
import time


nn = neural_network.NeuralNetwork((1, 28, 28), False)
nn.storeNetwork(1, "load")

print(nn.layers[0].kernels)


plt.imshow(nn.layers[0].kernels[1][0], cmap='gray')  # Use cmap='gray' for grayscale images
plt.title('Sample Image')
plt.colorbar()  # Display colorbar if needed
plt.show()