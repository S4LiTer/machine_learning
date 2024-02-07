import scipy.signal as sp
import matplotlib.pyplot as plt
import numpy as np
import math

class conv_types:
    same = "same"
    valid = "valid"


class ConvolutionalLayer:

    def __init__(self, input_size: tuple, kernel_size: tuple, kernel_count: int, correlation_type: str, activation, optimizer = "None"):
        self.act = activation["function"]
        self.der_act = activation["derivation"]

        self.correlation_type = correlation_type
        self.optimizer = optimizer
        self.beta = 0.6

        self.input_size = input_size
        self.output_size = (kernel_count,) + self.calculate_output_size(input_size[1:], kernel_size, self.correlation_type)
        
        self.kernel_size = (input_size[0],) + kernel_size
        self.kernel_count = kernel_count

        limit = math.sqrt(2/input_size[0])
        self.kernels = np.random.normal(0, limit, (kernel_count,) + self.kernel_size)

        self.biases = np.zeros(self.output_size)

        self.kernels_M = np.zeros((kernel_count,) + self.kernel_size)
        self.biases_M = np.zeros(self.output_size)
 
        self.past_inputs = []
        self.past_z = []

        self.layer_data = {"layer_type": "convolutional", "output_size": self.output_size, 
                           "kernel_size": kernel_size, "kernel_count": kernel_count, 
                           "correlation_type": correlation_type,
                           "activation": activation["name"], "optimizer": optimizer}


    def storeValues(self, order: int, id: int, action: str, path: str):
        name_kernels = f"{path}{id}_{order}_k.npy"
        name_biases = f"{path}{id}_{order}_b.npy"
        
        if action == "save":
            np.save(name_kernels, self.kernels)
            np.save(name_biases, self.biases)
        else:
            self.kernels = np.load(name_kernels)
            self.biases = np.load(name_biases)


    def forward_pass(self, input_matrix: np.ndarray, save_inputs = False):
        
        if input_matrix.shape != self.input_size:
            print("[ERROR] invalid input shape to predict function")
            print(self.input_size, input_matrix.shape)
            exit

        z = self.calculate_z(input_matrix)

        if save_inputs:
            self.past_z.append(z)
            self.past_inputs.append(input_matrix)
            

        return self.act(z)

    
    def backward_pass(self, output_gradient_list, learning_rate: float):
        self.bp_biases = np.zeros(self.output_size)
        self.bp_kernels = np.zeros_like(self.kernels)
        input_gradients = np.zeros((output_gradient_list.shape[0],) + self.input_size)

        
        # iterates through all recived output gradients and connects then with stored inputs/z
        # it sums all calculated gradients from discrete samples
        for output_index, output_gradient in enumerate(output_gradient_list):
            z_gradient = np.multiply(self.past_z[output_index], output_gradient)

            self.bp_biases = np.add(self.bp_biases, z_gradient)

            kernel_index = 0

            while kernel_index < self.kernel_count:
                self.bp_kernels[kernel_index] = np.add(self.bp_kernels[kernel_index], self.calculate_kernel_gradient(z_gradient[kernel_index], kernel_index, output_index))
                
                input_gradients[output_index] = np.add(input_gradients[output_index], self.calculate_input_gradient(z_gradient[kernel_index], kernel_index))
                kernel_index += 1



            
        self.bp_biases = self.bp_biases/len(output_gradient_list)
        self.bp_kernels = self.bp_kernels/len(output_gradient_list)
        
        if self.optimizer == "RMSprop":
            self.RMSprop(learning_rate)
        else:
            self.gradient_descent(learning_rate)


        self.past_inputs = []
        self.past_z = []
        return input_gradients


    def calculate_kernel_gradient(self, z_gradient: np.ndarray, kernel_index: int, output_index: int) -> np.ndarray:
        padded_input = self.past_inputs[output_index]

        if self.correlation_type == conv_types.same:
            kernel_shape = self.kernel_size[1:]
            pad = ((0, 0),
                (math.ceil(kernel_shape[0]/2-1), math.floor(kernel_shape[0]/2)),
                (math.ceil(kernel_shape[1]/2-1), math.floor(kernel_shape[1]/2)))
            
            padded_input = np.pad(padded_input, pad)
        
        return self.correlate(padded_input, z_gradient, "valid")

    def calculate_input_gradient(self, z_gradient: np.ndarray, kernel_index: int) -> np.ndarray:
        gradient = np.zeros(self.input_size)

        convolution_type = "same"
        if self.correlation_type == "valid":
            convolution_type = "full"

        layer_index = 0
        for kernel_layer in self.kernels[kernel_index]:
            gradient[layer_index] = sp.convolve2d(z_gradient, kernel_layer, convolution_type)

            layer_index += 1

        return gradient


    def calculate_z(self, input_matrix: np.ndarray):
        """
        This function calculates z (result before activation function)
        Returns: 3D numpy array with size matching defined output_size
        """



        # defines 3D array with output dimesions
        z = np.empty(self.output_size)

        # iterates through kernels and adds result of correlation to corresponding position in z array
        kernel_index = 0
        for kernel in self.kernels:
            
            conv_result = self.correlate(input_matrix, kernel, self.correlation_type)

            # sums results of convolutions by one kernel over z axis to one 2D matrix which is stored to predefined z array
            z[kernel_index] = np.sum(conv_result, axis=0)


            kernel_index += 1

        z = np.add(z, self.biases)
        return z

    def correlate(self, input_matrix: np.ndarray, kernel: np.ndarray, correlation_type: str):
        """
        This function is used to correlate two equally deep matrixes (kernel and input_matrix)
        Returns: 3D numpy array with same depth and width and height defined by correlation
        """

        kernel_shape = kernel.shape
        if len(kernel_shape) == 3:
            kernel_shape = kernel_shape[1:]
        

        result_size = (input_matrix.shape[0],) + self.calculate_output_size(input_matrix.shape[1:], kernel_shape, correlation_type)


        # defines 3D array for one kernel correlation output (z axis is defined by input depth)
        conv_result = np.empty(result_size)
        index = 0
        kernel_depth = 0
        
        # iterates through all inputs
        while index < self.input_size[0]:

            used_kernel = None
            if len(kernel.shape) == 3:
                used_kernel = kernel[index]
            else:
                used_kernel = kernel

            conv_result[index] = sp.correlate2d(input_matrix[index], used_kernel, correlation_type)
            index += 1

        return conv_result

    def calculate_output_size(self, input_size: tuple, used_kernel_size: tuple, correlation_type: str) -> tuple:
        """
        Calculates matrix size after correlation
        Reuturns: tuple with 2D size of matrix after correlation.

        Depth does not depend on correlation
        """

        if correlation_type == conv_types.same:
            return (input_size[0], input_size[1])
        
        elif correlation_type == conv_types.valid:
            return (input_size[0] - used_kernel_size[0] + 1, input_size[1] - used_kernel_size[1] + 1)

        else:
            print("[ERROR] invalid correlation type:", self.correlation_type)



##### DESCENT FUNCTIONS
    def gradient_descent(self, learning_rate: float):
        self.weights = np.subtract(self.kernels, self.bp_kernels*learning_rate)
        self.biases = np.subtract(self.biases, self.bp_biases*learning_rate)


    def RMSprop(self, learning_rate: float):
        self.biases_M = np.add(self.beta*self.biases_M, (1-self.beta)* np.power(self.bp_biases, 2) )

        mlt = learning_rate/(np.sqrt(self.biases_M.copy()) + 0.00001)
        mlt = np.multiply(mlt, self.bp_biases)

        self.biases = np.subtract(self.biases, mlt)


        self.kernels_M = np.add(self.beta*self.kernels_M, (1-self.beta)* np.power(self.bp_kernels, 2))


        mlt = learning_rate/(np.sqrt(self.kernels_M.copy()) + 0.00001)
        mlt = np.multiply(mlt, self.bp_kernels)
        
        self.kernels = np.subtract(self.kernels, mlt)
        
        """
        fig, axs = plt.subplots(3, 3, figsize=(10, 3))

        # Display each image on a subplot
        axs[0][0].imshow(self.bp_kernels[0][0], cmap='gray')
        axs[0][0].set_title('Image 1')

        axs[0][1].imshow(self.bp_kernels[1][0], cmap='gray')
        axs[0][1].set_title('Image 2')

        axs[0][2].imshow(self.bp_kernels[2][0], cmap='gray')
        axs[0][2].set_title('Image 3')

        axs[1][0].imshow(self.kernels[0][0], cmap='gray')
        axs[1][0].set_title('Imag 1')

        axs[1][1].imshow(self.kernels[1][0], cmap='gray')
        axs[1][1].set_title('Imag 2')

        axs[1][2].imshow(self.kernels[2][0], cmap='gray')
        axs[1][2].set_title('Imag 3')

        # Display each image on a subplot
        axs[2][0].imshow(mlt[0][0], cmap='gray')
        axs[2][0].set_title('mlt 1')

        axs[2][1].imshow(mlt[1][0], cmap='gray')
        axs[2][1].set_title('mlt 2')

        axs[2][2].imshow(mlt[2][0], cmap='gray')
        axs[2][2].set_title('mlt 3')

        plt.show()
        """




