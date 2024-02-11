import numpy as np
import matplotlib.pyplot as plt
import time


class FlatteningLayer:
    def __init__(self, input_size: tuple) -> None:
        
        self.input_size = input_size

        self.output_size = 1
        for dimesion in input_size:
            self.output_size *= dimesion


        self.layer_data = {"layer_type": "flattening", "output_size": self.output_size}

    
    def storeValues(self, order, id, action, path):
        """
        there are no values but this function still has to be here because it is called from NeuralNetowrk class
        """
        
        return
    

    def forward_pass(self, input_matrix: np.ndarray, save_inputs=False) -> np.ndarray:
        if len(input_matrix.shape) == len(self.input_size):
            return input_matrix.reshape((self.output_size))
        
        elif len(input_matrix.shape) == len(self.input_size) + 1:
            return input_matrix.reshape((input_matrix.shape[0], self.output_size))
        
        else:
            print("[ERROR] Invalid input to flattening layer")
    
    def backward_pass(self, output_gradient_list, learning_rate=0):
        return output_gradient_list.reshape((output_gradient_list.shape[0],) + self.input_size)
        