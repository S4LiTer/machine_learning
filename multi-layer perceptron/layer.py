import numpy as np
import math
import time
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, input_neuron, output_neurons, activation_func, derivation_activation):
        self.act = activation_func
        self.der_act = derivation_activation

        self.weights = np.random.random((output_neurons, input_neuron))/(input_neuron**0.5)
        self.biases = np.random.random((output_neurons))
        self.all_inputs = []

        self.qqq = 0


    def storeValues(self, order, id, action, path):
        name_weights = f"{path}w_{self.weights.shape[0]}_{self.weights.shape[1]}_{order}_{id}"
        name_biases = f"{path}b_{self.weights.shape[0]}_{self.weights.shape[1]}_{order}_{id}"
        if action == "save":
            np.savetxt(name_weights, self.weights, delimiter=',')
            np.savetxt(name_biases, self.biases, delimiter=',')
        else:
            self.weights = np.loadtxt(name_weights, delimiter=',')
            self.biases = np.loadtxt(name_biases, delimiter=',')

    

    def predict(self, input_neurons, save_inputs = False):
        if save_inputs:
            self.all_inputs.append(input_neurons)
            
        z = self.calculate_z(input_neurons, flip=False)
        return self.act(z)
    

    def adjust(self, output_gradient_list, learning_rate: float):
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

            biases_gradient = np.multiply(z, output_gradient)
            self.bp_biases = np.add(self.bp_biases, biases_gradient)


            prod = np.multiply(z, output_gradient)[:, None]
            input_gradient = np.multiply(self.weights, prod)
            input_gradient = np.sum(input_gradient, axis=0)
            input_gradients.append(input_gradient)


        self.bp_weights = self.bp_weights/len(output_gradient_list)
        self.weights = np.subtract(self.weights, self.bp_weights*learning_rate)

        

        self.bp_biases = self.bp_biases/len(output_gradient_list)
        self.biases = np.subtract(self.biases, self.bp_biases*learning_rate)

        self.all_inputs = []
        return input_gradients


    def calculate_z(self, prev_layer, flip=False):
        # sketchy AH sektor...
        product = np.dot(self.weights, prev_layer)
        z = np.add(product, self.biases)
        
        if flip:
            z = z[:, None] # converts z to column

        return z
        

"""
activation = lambda x: 1/(1+(math.e**-x))
d_activation = lambda x: activation(x)*(1-activation(x))

l = Layer(3, 2, activation, d_activation)
print("prediction:", l.predict(  np.array([1, 2, 3])  ))
print("-"*40)
l.adjust([np.array([0.5, 1]), np.array([1, 2]), np.array([0.5, 0])], 1)
"""
