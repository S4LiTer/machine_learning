import numpy as np
import math
import time


class pooling_types:
    max_pooling = "max"
    average_pooling = "avg"


class PoolingLayer:
    def __init__(self, input_size: np.ndarray, pooling_size, pooling_type=pooling_types.max_pooling):
        self.input_size = input_size
        self.pool_size = pooling_size
        self.pooling_type = pooling_type
        self.output_size = (input_size[0], self.input_size[1] // self.pool_size[0], self.input_size[2] // self.pool_size[1])

        self.stride = pooling_size


    def forward_pass(self, input_matrix, save_inputs=False):
        if self.stride == self.pool_size:
            pooled_window = input_matrix[:self.output_size[0], :self.output_size[1]*self.pool_size[0], :self.output_size[2]*self.pool_size[1]]
            reshaped_array = pooled_window.reshape((self.output_size[0], self.output_size[1], self.pool_size[0], self.output_size[2], self.pool_size[1]))
            if self.pooling_type == pooling_types.max_pooling:
                return np.max(reshaped_array, axis=(2, 4))
            else:
                return np.mean(reshaped_array, axis=(2, 4))

