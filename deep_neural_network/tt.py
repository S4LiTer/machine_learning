import neural_network
import layers
import matplotlib.pyplot as plt
from utils import activation_functions
import numpy as np

res = np.array([[0.1, 0.2, 0.8],
                [0.2, 0.5, 0.2],
                [0.1, 0.1, 0.8],
                [0.4, 0.2, 0.8],
                [0.1, 1, 0.8]])

y = np.array([[0, 0, 1],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 1],
              [1, 0, 0]])

matches = np.equal(np.argmax(res, axis=1), np.argmax(y, axis=1))
accuracy = np.mean(matches) * 100

print("Accuracy: {:.2f}%".format(accuracy))