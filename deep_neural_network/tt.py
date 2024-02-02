from utils import activation_functions as act_funcs
from utils import display
import preprocessing
import numpy as np
import layers
import time


conv = layers.ConvolutionalLayer((3, 3), 1, 2, 1, layers.conv_types.valid, act_funcs.functions["relu"])

random_input = np.round(np.random.rand(1, 3, 3))

print(conv.predict(random_input))