import numpy as np
import math
import time

class FullyConnectedLayer:
    def __init__(self, input_neurons: int, output_neurons: int, activation, optimizer: str):
        self.act = activation["function"]
        self.der_act = activation["derivation"]

        self.output_size = output_neurons

        limit = math.sqrt(6.0/float(input_neurons))
        self.weights = np.random.uniform(low=-limit, high=limit, size=(output_neurons, input_neurons))
        self.biases = np.random.random((output_neurons))

        self.weights_M = np.zeros((output_neurons, input_neurons))
        self.biases_M = np.zeros((output_neurons))


        self.all_inputs = []
        self.optimizer = optimizer

        self.layer_data = {"layer_type": "fully_connected", "output_size": output_neurons, "activation": activation["name"], "optimizer": optimizer}

        # for RMSprop optimizer
        self.beta = 0.9


    def storeValues(self, order: int, id: int, action: str, path: str):
        name_weights = f"{path}{id}_{order}_w.npy"
        name_biases = f"{path}{id}_{order}_b.npy"
        
        if action == "save":
            np.save(name_weights, self.weights)
            np.save(name_biases, self.biases)
        else:
            self.weights = np.load(name_weights)
            self.biases = np.load(name_biases)

    

    def forward_pass(self, input_neurons: np.ndarray, save_inputs = False) -> np.ndarray:
        if save_inputs:
            self.all_inputs.append(input_neurons)
            
        z = self.calculate_z(input_neurons, flip=False)
        return self.act(z)
    

    def backward_pass(self, output_gradient_list, learning_rate: float):
        self.bp_weights = np.zeros_like(self.weights)
        self.bp_biases = np.zeros_like(self.biases)
        input_gradients = []
        input_index = 0

        
        for output_gradient in output_gradient_list:
            last_input = self.all_inputs[input_index]
            input_index += 1

            z = self.calculate_z(last_input, flip=True)
            z = self.der_act(z)

            weight_gradient = np.multiply(last_input, z)
            weight_gradient = np.multiply(weight_gradient, output_gradient[:, None])
            self.bp_weights = np.add(self.bp_weights, weight_gradient)

            
            z = self.calculate_z(last_input, flip=False)
            z = self.der_act(z)
            # maybe replace with 
            # z = z[None, :]
            
            biases_gradient = np.multiply(z, output_gradient)
            self.bp_biases = np.add(self.bp_biases, biases_gradient)


            prod = np.multiply(z, output_gradient)[:, None]
            input_gradient = np.multiply(self.weights, prod)
            input_gradient = np.sum(input_gradient, axis=0)
            input_gradients.append(input_gradient)


        self.bp_weights = self.bp_weights/len(output_gradient_list)
        self.bp_biases = self.bp_biases/len(output_gradient_list)

        if self.optimizer == "RMSprop":
            self.RMSprop(learning_rate)
        else:
            self.gradient_descent(learning_rate)


        self.all_inputs = []
        return input_gradients


    def calculate_z(self, prev_layer: np.ndarray, flip=False) -> np.ndarray:
        # sketchy AH sektor...
        product = np.dot(self.weights, prev_layer)
        z = np.add(product, self.biases)
        
        if flip:
            z = z[:, None] # converts z to column

        return z
        


##### DESCENT FUNCTIONS


    def gradient_descent(self, learning_rate: float):
        self.weights = np.subtract(self.weights, self.bp_weights*learning_rate)
        self.biases = np.subtract(self.biases, self.bp_biases*learning_rate)

    
    def RMSprop(self, learning_rate: float):
        self.weights_M = np.add(self.beta*self.weights_M, (1-self.beta)* np.power(self.bp_weights, 2))

        non_zero_M = self.weights_M.copy()
        non_zero_M[non_zero_M == 0] = 0.0001
        
        mlt = learning_rate/np.sqrt(non_zero_M)
        mlt = np.multiply(mlt, self.bp_weights)

        self.weights = np.subtract(self.weights, mlt)



        self.biases_M = np.add(self.beta*self.biases_M, (1-self.beta)* np.power(self.bp_biases, 2) )

        non_zero_M = self.biases_M.copy()
        non_zero_M[non_zero_M == 0] = 0.0001
        
        mlt = learning_rate/np.sqrt(non_zero_M)
        mlt = np.multiply(mlt, self.bp_biases)

        self.biases = np.subtract(self.biases, mlt)
 
