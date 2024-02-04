from utils import activation_functions as act_funcs
import matplotlib.pyplot as plt
import neural_network
from mnist import MNIST
import numpy as np
import layers
import json
import time


"""
test_input = np.round(np.random.random((3, 6, 6)) * 20)

pool = layers.PoolingLayer((3, 6, 6), (2, 2))
res = pool.forward_pass(test_input)
print(test_input)
print(res)

start_time = time.time()
mndata = MNIST("samples")

_testing_images, _testing_labels = mndata.load_testing()
testing_images = np.array(_testing_images)/255

image = testing_images[4].reshape((1, 28, 28))

print("TIME TO LOAD MNIST: ", time.time() - start_time, "\n\n")

conv1 = layers.ConvolutionalLayer((1, 28, 28), (4, 4), 3, layers.conv_types.same, act_funcs.functions["relu"], "RMSprop")
conv2 = layers.ConvolutionalLayer((3, 28, 28), (4, 4), 5, layers.conv_types.same, act_funcs.functions["relu"], "RMSprop")


res = conv1.predict(image, save_inputs=True)
res = conv2.predict(res, save_inputs=True)


test_gradient = [np.random.rand(5, 28, 28)]

bc = conv2.adjust(test_gradient, 1)

conv2.storeValues(0, 0, "save", "")




fig, axs = plt.subplots(1, 4)


axs[0].imshow(image[0], cmap='gray')
axs[0].set_title('INPUT')

axs[1].imshow(res[0], cmap='gray')
axs[1].set_title('RESULT 2')

axs[2].imshow(res[1], cmap='gray')
axs[2].set_title('RESULT 2')

axs[3].imshow(res[2], cmap='gray')
axs[3].set_title('RESULT 3')

plt.tight_layout()

plt.show()
"""