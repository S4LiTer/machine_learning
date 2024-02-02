import scipy.signal as sp
import numpy as np
import math
import time

class conv_types:
    same = "same"
    valid = "valid"


class ConvolutionalLayer:
    def __init__(self, input_size: tuple, depth: int, kernel_size: int, kernel_count: int, correlation_type: str, activation):
        self.act = activation["function"]
        self.der_act = activation["derivation"]

        self.correlation_type = correlation_type

        self.input_size = (depth, input_size[0], input_size[1])

        self.output_size = self.calculate_output_size(input_size, kernel_size, kernel_count)

        self.kernels = [np.round(np.random.rand(depth, kernel_size, kernel_size)*2) for _ in range(kernel_count)]

        self.biases = np.random.random(self.output_size)
 
        self.past_inputs = []
        self.past_z = []


    def predict(self, input_matrix, save_inputs = False):
        if input_matrix.shape != self.input_size:
            print("[ERROR] invalid input shape to predict function")
            print(self.input_size, input_matrix.shape)
            exit

        z = self.calculate_z(input_matrix)

        if save_inputs:
            self.past_z.append(z)
            self.past_inputs.append(input_matrix)

        return self.act(z)

        
    def calculate_z(self, input_matrix):
        # defines 3D array with output dimesions
        z = np.empty(self.output_size)

        # iterates through kernels and adds result of correlation to corresponding position in z array
        kernel_index = 0
        for kernel in self.kernels:
            
            # defines 3D array for one kernel correlation output (z axis is defined by input depth)
            conv_result = np.empty((self.input_size[0], self.output_size[1], self.output_size[2]))
            index = 0
            while index < self.input_size[0]:
                conv_result[index] = sp.correlate2d(input_matrix[index], kernel[index], self.correlation_type)

                index += 1

            # sums results of convolutions by one kernel over z axis to one 2D matrix which is stored to predefined z array
            z[kernel_index] = np.sum(conv_result, axis=0)


            kernel_index += 1

        z = np.add(z, self.biases)
        return z



    def calculate_output_size(self, input_size, kernel_size, kernel_count):
        if self.correlation_type == conv_types.same:
            return (kernel_count, input_size[0], input_size[1])
        
        elif self.correlation_type == conv_types.valid:
            return (kernel_count, input_size[0] - kernel_size + 1, input_size[1] - kernel_size + 1)

        else:
            print("[ERROR] invalid correlation type:", self.correlation_type)


"""
ConvolutionalLayer((1, 1), 1)

random_input = np.round(np.random.rand(6, 28, 28) * 5)

kernel = np.random.rand(6, 2, 2)

print(random_input)
print(kernel)

index = 0

result = np.ndarray((6, 28, 28))
for layer in random_input:
    print(index)
    result[index] = sp.correlate2d(layer, kernel[index], "same")
    index += 1

print(result)
print(np.sum(result, axis=0))
"""