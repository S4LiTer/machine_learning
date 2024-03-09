import numpy as np
import time


class pooling_types:
    max_pooling = "max"
    average_pooling = "avg"


class PoolingLayer:
    def __init__(self, input_size: np.ndarray, pooling_size: tuple, pooling_type=pooling_types.max_pooling):
        self.input_size = input_size
        self.pool_size = pooling_size
        self.pooling_type = pooling_type
        self.output_size = (input_size[0], self.input_size[1] // self.pool_size[0], self.input_size[2] // self.pool_size[1])

        self.pooled_matrix_size = (input_size[0], self.output_size[1]*self.pool_size[0], self.output_size[2]*self.pool_size[1])

        self.indicies = np.array([])

        self.layer_data = {"layer_type": "pooling", "output_size": self.output_size, "pool_size": self.pool_size, "pooling_type": pooling_type}

    def storeValues(self, order, id, action, path):
        """
        there are no values but this function still has to be here because it is called from NeuralNetowrk class
        """
        
        return

    def forward_pass(self, input_matrix: np.ndarray, save_inputs=False):
        if len(input_matrix.shape) == len(self.input_size):
            input_matrix = input_matrix.reshape((1,) + self.input_size)
        batch_count = input_matrix.shape[0]

        pooled_window = input_matrix[:batch_count, :self.pooled_matrix_size[0], :self.pooled_matrix_size[1], :self.pooled_matrix_size[2]]
        
        # Reshapes the array to 5D array where are cubes of size self.pool_size[0], self.output_size[2], self.pool_size[1]
        # I am then able to find max value just on top side of the cube (along first and third axis)
        # I cannot form squares, because then the square would be formed from one row and not one square according to pooling window
        reshaped_array = pooled_window.reshape((batch_count, self.output_size[0], self.output_size[1], self.pool_size[0], self.output_size[2], self.pool_size[1]))
    
        if self.pooling_type == pooling_types.average_pooling:
            return np.mean(reshaped_array, axis=(3, 5))
        
        
        result = np.max(reshaped_array, axis=(3, 5))

        if save_inputs:
            # so I am able to put them in row and take argmax of it
            # Rotates the cubes and moves values from the top to the front 
            # => Numbers are now sorted according to sliding pooling window
            rotated_array = np.rot90(reshaped_array, k=1, axes=(4, 3))[:, :, :, :, ::-1, :]
            rotated_array = rotated_array.reshape((batch_count, ) + self.output_size + (self.pool_size[0]*self.pool_size[1],))
            indicies = np.argmax(rotated_array, axis=-1)

            self.indicies = np.append(self.indicies, indicies)


        return result
        
    def backward_pass(self, output_gradient_list, learning_rate:float):
        if output_gradient_list[0].shape != self.output_size:
            print("[ERROR] Invalid input to backward pass")
            return
        
    
        grad_shape = output_gradient_list.shape

        output_gradient_list = output_gradient_list.reshape((-1))
        input_gradient = np.zeros((output_gradient_list.shape[0], self.pool_size[0]*self.pool_size[1]))

        for index, grad in enumerate(output_gradient_list):
            if self.pooling_type == pooling_types.max_pooling:
                input_gradient[index, int(self.indicies[index])] = grad
            else:
                input_gradient[index] = input_gradient[index] + grad/(self.pool_size[0]*self.pool_size[1])


        input_gradient = input_gradient.reshape(grad_shape + self.pool_size)
        input_gradient = input_gradient[:, :, :, :, ::-1, :]
        input_gradient = np.rot90(input_gradient, k=-1, axes=(4, 3))
        input_gradient = input_gradient.reshape((grad_shape[0], ) + self.pooled_matrix_size)

        if input_gradient.shape[1:] != self.input_size:
            pad = (self.input_size[1] - input_gradient.shape[2], 
                   self.input_size[2] - input_gradient.shape[3])
            input_gradient = np.pad(input_gradient, ((0, 0), (0, 0), (0, pad[0]), (0, pad[1])))
            
        self.indicies = np.array([])
        return input_gradient
    

    
