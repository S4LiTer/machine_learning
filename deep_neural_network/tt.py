import neural_network
import layers
import matplotlib.pyplot as plt
from utils import activation_functions
import numpy as np

fcl = layers.FullyConnectedLayer(80, 10, activation=activation_functions.functions["sigmoid"], optimizer="")

test = np.random.random((80))

fcl.forward_pass(test, save_inputs=True)

test = np.random.random((10))

print(test.shape)
fcl.backward_pass(test, 0.1)
