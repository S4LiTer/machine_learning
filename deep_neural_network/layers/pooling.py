import numpy as np


class pooling_types:
    max_pooling = "max"
    average_pooling = "avg"


class PoolingLayer:
    def __init__(self, input_size: np.ndarray, pooling_size: tuple, pooling_type=pooling_types.max_pooling):
        self.input_size = input_size
        self.pool_size = pooling_size
        self.pooling_type = pooling_type
        self.output_size = (input_size[0], self.input_size[1] // self.pool_size[0], self.input_size[2] // self.pool_size[1])

        self.stride = pooling_size

        self.indicies = []

        self.layer_data = {"layer_type": "pooling", "output_size": self.output_size, "pool_size": self.pool_size, "pooling_type": pooling_type}

    def storeValues(self, order, id, action, path):
        """
        there are no values but this function still has to be here because it is called from NeuralNetowrk class
        """
        
        return

    def forward_pass(self, input_matrix: np.ndarray, save_inputs=False):

        if self.stride == self.pool_size:

            pooled_window = input_matrix[:self.output_size[0], :self.output_size[1]*self.pool_size[0], :self.output_size[2]*self.pool_size[1]]
            
            # Reshapes the array to 5D array where are cubes of size self.pool_size[0], self.output_size[2], self.pool_size[1]
            # I am then able to find max value just on top side of the cube (along first and third axis)
            # I cannot form squares, because then the square would be formed from one row and not one square according to pooling window
            reshaped_array = pooled_window.reshape((self.output_size[0], self.output_size[1], self.pool_size[0], self.output_size[2], self.pool_size[1]))
        
            if self.pooling_type == pooling_types.average_pooling:
                return np.mean(reshaped_array, axis=(2, 4))
            
            
            result = np.max(reshaped_array, axis=(2, 4))

            if save_inputs:
                # so I am able to put them in row and take argmax of it
                # Rotates the cubes and moves values from the top to the front 
                # => Numbers are now sorted according to sliding pooling window
                rotated_array = np.rot90(reshaped_array, k=1, axes=(3, 2))[:, :, :, ::-1, :]
                rotated_array = rotated_array.reshape(self.output_size + (self.pool_size[0]*self.pool_size[1],))
                indicies = np.argmax(rotated_array, axis=-1)

                self.indicies.append(indicies)


            return result
        
    def backward_pass(self, output_gradient_list, learning_rate:float):
        if output_gradient_list[0].shape != self.output_size:
            print("[ERROR] Invalid input to backward pass")
            return
        

        input_gradients = np.zeros((output_gradient_list.shape[0],) + (self.input_size))

        for output_index, output_gradient in enumerate(output_gradient_list):
            input_gradient = np.zeros(self.input_size)

            for channel_index, channel in enumerate(output_gradient):
                for column_index, column in enumerate(channel):
                    for row_index, row in enumerate(column):
                        if self.pooling_type == pooling_types.average_pooling:
                            first_pos = (column_index*self.pool_size[0], row_index*self.pool_size[1])
                            last_pos = (first_pos[0] + self.pool_size[0], first_pos[1] + self.pool_size[1])

                            input_gradient[channel_index, first_pos[0]:last_pos[0], first_pos[1]:last_pos[1]] = row/(self.pool_size[0] * self.pool_size[1])
                        else:
                            grad_pos = np.unravel_index(self.indicies[output_index][channel_index][column_index][row_index], self.pool_size)
                            grad_pos = (channel_index, grad_pos[0]+column_index*self.pool_size[0], grad_pos[1]+row_index*self.pool_size[1])
                            input_gradient[grad_pos] = row
            


            input_gradients[output_index] = input_gradient


        return input_gradients
    
