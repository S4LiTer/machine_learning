import numpy as np
import math
import time

class FullyConnectedLayer:
    def __init__(self, input_neurons: int, output_neurons: int, activation, optimizer: str):
        self.act = activation["function"]
        self.der_act = activation["derivation"]

        self.output_size = output_neurons
        self.input_size = input_neurons

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
        # Uloží hodnoty vah a vychýlení
        name_weights = f"{path}{id}_{order}_w.npy"
        name_biases = f"{path}{id}_{order}_b.npy"
        
        if action == "save":
            np.save(name_weights, self.weights)
            np.save(name_biases, self.biases)
        else:
            self.weights = np.load(name_weights)
            self.biases = np.load(name_biases)

    

    def forward_pass(self, input_values: np.ndarray, save_inputs = False) -> np.ndarray:
        # Dopředný průchod. Vypočítá výsledek vrtvy podle vstupu
        # Funkce dokáže pracovat s daty více vzorků najednou.

        z = self.calculate_z(input_values)

        # Uloží vstupy a výstup před aktivační funkcí pro použití při učení
        if save_inputs:
            self.past_z = z
            self.all_inputs = input_values

            if len(self.past_z.shape) == 1:
                self.past_z = self.past_z.reshape((-1, self.past_z.shape[0]))
                self.all_inputs = self.all_inputs.reshape((-1, self.all_inputs.shape[0]))



        result = self.act(z)

        if len(result.shape) == 2 and result.shape[0] == 1:
            result = result.reshape(-1)
        return result

    

    def backward_pass(self, output_gradient_list, learning_rate: float):
        # Zpětné šíření chyby -> upraví hodnoty kernelů a vychýlení

        # předdefinuje matice pro gradienty
        self.bp_weights = np.zeros_like(self.weights)
        self.bp_biases = np.zeros_like(self.biases)
        input_gradients = np.zeros((output_gradient_list.shape[0], self.input_size))


        # Projde gradient pro všechny vzorky a podle nich vypočítá gradient vah a vychýlení
        for input_index, output_gradient in enumerate(output_gradient_list):
            last_input = self.all_inputs[input_index]
            z = None

            # Pokud je toto výstupní vrsta (používá softmax) tak je output_gradient shodný s gradientem výsledku před aktivační funkcí
            if self.layer_data["activation"] != "softmax":
                z = self.past_z[input_index]
                z = self.der_act(z)
                z= np.multiply(z, output_gradient)
            else:
                z = output_gradient

            weight_gradient = np.multiply(last_input, z[None, :].T)
            self.bp_weights = np.add(self.bp_weights, weight_gradient)

            self.bp_biases = np.add(self.bp_biases, z)

            input_gradients[input_index] = np.dot(self.weights.T, z)


        # Zprůměruje gradienty a pomocí gradientního sestupu upraví jejich hodnoty
        self.bp_weights = self.bp_weights/len(output_gradient_list)
        self.bp_biases = self.bp_biases/len(output_gradient_list)

        self.all_inputs = []
        self.past_z = []

        if self.optimizer == "RMSprop":
            self.RMSprop(learning_rate)
        else:
            self.gradient_descent(learning_rate)

        return input_gradients


    def calculate_z(self, prev_layer: np.ndarray) -> np.ndarray:
        # Vypočítá výsledek vrstvy před aktivační funkcí

        if len(prev_layer.shape) == 1:
            product = np.dot(self.weights, prev_layer)
            z = np.add(product, self.biases)
            return z
        elif len(prev_layer.shape) == 2:
            product = np.dot(prev_layer, self.weights.T)
            z = np.add(product, self.biases)
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
 
