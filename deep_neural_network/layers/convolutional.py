import scipy.signal as sp
import matplotlib.pyplot as plt
import numpy as np
import time
import math

class conv_types:
    same = "same"
    valid = "valid"


class ConvolutionalLayer:

    def __init__(self, input_size: tuple, kernel_size: tuple, kernel_count: int, correlation_type: str, activation, optimizer = "None"):
        self.act = activation["function"]
        self.der_act = activation["derivation"]

        self.correlation_type = correlation_type

        self.input_size = input_size
        self.output_size = (kernel_count,) + self.calculate_output_size(input_size[1:], kernel_size, self.correlation_type)
        
        self.kernel_size = (input_size[0],) + kernel_size
        self.kernel_count = kernel_count

        limit = math.sqrt(2/(self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]))
        self.kernels = np.random.normal(0, limit, (kernel_count,) + self.kernel_size)

        self.biases = np.random.random(self.output_size)

        self.kernels_M = np.zeros((kernel_count,) + self.kernel_size)
        self.biases_M = np.zeros(self.output_size)
 
        self.past_inputs = np.array([])
        self.past_z = np.array([])

        self.optimizer = optimizer
        self.beta = 0.85

        self.layer_data = {"layer_type": "convolutional", "output_size": self.output_size, 
                           "kernel_size": kernel_size, "kernel_count": kernel_count, 
                           "correlation_type": correlation_type,
                           "activation": activation["name"], "optimizer": optimizer}


    def storeValues(self, order: int, id: int, action: str, path: str):
        # Načte hodnoty vychýlení a kernelů

        name_kernels = f"{path}{id}_{order}_k.npy"
        name_biases = f"{path}{id}_{order}_b.npy"
        
        if action == "save":
            np.save(name_kernels, self.kernels)
            np.save(name_biases, self.biases)
        else:
            self.kernels = np.load(name_kernels)
            self.biases = np.load(name_biases)


    def forward_pass(self, input_matrix: np.ndarray, save_inputs = False):
        # Provene konvoluci

        # kontrola vstupních rozměrů
        if len(input_matrix.shape) == len(self.input_size):
            input_matrix = input_matrix.reshape((1,) + self.input_size)

        if input_matrix.shape[1:] != self.input_size:
            print(f"[ERROR] invalid input shape to predict function (expected: {self.input_size}, given: {input_matrix.shape[1:]})")


        # Předdefinuje matici pro výstup
        output = np.zeros((input_matrix.shape[0], ) + self.output_size)

        if save_inputs:
            # Uloží výstupní hodnoty před aktivačí funkcí a vstupy 
            self.past_z = np.zeros((input_matrix.shape[0], ) + self.output_size)
            self.past_inputs = input_matrix


        # Provede pro každý kanál zvlášť korelaci a výsledek nechá projít aktivační funkcí
        for index, input_mat in enumerate(input_matrix):
            z = self.calculate_z(input_mat)

            if save_inputs:
                self.past_z[index] = z
                
            output[index] = self.act(z)

        return output

    
    def backward_pass(self, output_gradient_list, learning_rate: float):
        # Zpětné šíření chyby -> upraví hodnoty kernelů a vychýlení

        self.bp_biases = np.zeros(self.output_size)
        self.bp_kernels = np.zeros_like(self.kernels)
        input_gradients = np.zeros((output_gradient_list.shape[0],) + self.input_size)

        # Projde všechny kanály gradientů
        for output_index, output_gradient in enumerate(output_gradient_list):
            z_gradient = np.multiply(self.der_act(self.past_z[output_index]), output_gradient)

            # Gradint vychýlení je stejný jako gradient před aktivační funkcí
            self.bp_biases = np.add(self.bp_biases, z_gradient)

            kernel_index = 0

            # Pro každý kernel vypočte gradient hodnot v kernelu a gradient vstupů
            while kernel_index < self.kernel_count:
                kernel_gradient = self.calculate_kernel_gradient(z_gradient[kernel_index], output_index)
                self.bp_kernels[kernel_index] = np.add(self.bp_kernels[kernel_index], kernel_gradient)

                input_gradient = self.calculate_input_gradient(z_gradient[kernel_index], kernel_index)
                input_gradients[output_index] = np.add(input_gradients[output_index], input_gradient)
                kernel_index += 1



        # Zprůměruje gradienty kernelů a vychýlení a následné pomocí gradientního sestupu upraví jejich hodnoty
        self.bp_biases = self.bp_biases/len(output_gradient_list)
        self.bp_kernels = self.bp_kernels/len(output_gradient_list)
        
        if self.optimizer == "RMSprop":
            self.RMSprop(learning_rate)
        else:
            self.gradient_descent(learning_rate)


        # Vymeže uložené data
        self.past_inputs = np.array([])
        self.past_z = np.array([])

        return input_gradients


    def calculate_kernel_gradient(self, z_gradient: np.ndarray, output_index: int) -> np.ndarray:
        # Funkce slouží pro výpočet gradientu kernelu
        # Gradient kernelu můžeme dostat pokud provedeme korelaci gradientu výsledku vrstvy (před aktivační funkcí) po vstupu (gradient je pohybující se kernel). 
        # Pokud používáme same kovoluci, musíme přidat padding, abychom zachovali velikost kernelu

        padded_input = self.past_inputs[output_index]

        if self.correlation_type == conv_types.same:
            kernel_shape = self.kernel_size[1:]
            pad = ((0, 0),
                (math.ceil(kernel_shape[0]/2-1), math.floor(kernel_shape[0]/2)),
                (math.ceil(kernel_shape[1]/2-1), math.floor(kernel_shape[1]/2)))
            
            padded_input = np.pad(padded_input, pad)
        
        return self.correlate(padded_input, z_gradient, "valid")

    def calculate_input_gradient(self, z_gradient: np.ndarray, kernel_index: int) -> np.ndarray:
        # Vypočítá gradient vstupu pomocí konvoluce (kernel je otočen a 180 stupnu)

        gradient = np.zeros(self.input_size)

        # Abychom zachovali velikost vstupu musíme správně určit typ konvoluce
        convolution_type = "same"
        if self.correlation_type == "valid":
            convolution_type = "full"

        layer_index = 0
        # Pro každou matici v kernelu provede konvoluci přes gradient výstupu před aktivační funkčí
        # Protože jeden kernel má stejný počet kanalu jako je vstupních matic, provedeme konvoluci tolikrát jako je vstupních kanalu
        # takže budeme mít gradient pro každý kanal
        for kernel_layer in self.kernels[kernel_index]:
            gradient[layer_index] = sp.convolve2d(z_gradient, kernel_layer, convolution_type)

            layer_index += 1

        return gradient


    def calculate_z(self, input_matrix: np.ndarray):
        # Vypočítá výsledek sítě před aktivační funkcí


        # předdefinuje matici výsledků
        z = np.empty(self.output_size)

        # Projde všechny kernely. Každý kernel vytvoří jeden výstupní kanál
        kernel_index = 0
        for kernel in self.kernels:
            
            conv_result = self.correlate(input_matrix, kernel, self.correlation_type)

            # Sečte výsledek korelací ze všech kanálů v jednom kernelu
            z[kernel_index] = np.sum(conv_result, axis=0)


            kernel_index += 1

        # přičte vychýlení k výsledku
        z = np.add(z, self.biases)

        return z

    def correlate(self, input_matrix: np.ndarray, kernel: np.ndarray, correlation_type: str):
        # Tato funkce vypočítá korolaci dvou matic.
        # Pokud má kernel stejný počet kanálů jako input_matrix, 
        # bude počet výstupních matic shodný s počtem kanálů

        # Pokud má input_matrix více kanálů a kernel pouze jeden, provede se korelace kernelu po všech kanálech vstupní matice

        kernel_shape = kernel.shape
        if len(kernel_shape) == 3:
            kernel_shape = kernel_shape[1:]
        

        result_size = (input_matrix.shape[0],) + self.calculate_output_size(input_matrix.shape[1:], kernel_shape, correlation_type)


        conv_result = np.empty(result_size)
        index = 0
        
        # Projde všechny vstupy a provede na nich korelaci
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
        # Vypočítá velikost výstupu z vrstvy

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
        # Provede RMSprop podle matematického vzorce

        self.biases_M = np.add(self.beta*self.biases_M, (1-self.beta)* np.power(self.bp_biases, 2) )

        mlt = learning_rate/(np.sqrt(self.biases_M.copy()) + 0.00001)
        mlt = np.multiply(mlt, self.bp_biases)

        self.biases = np.subtract(self.biases, mlt)


        self.kernels_M = np.add(self.beta*self.kernels_M, (1-self.beta)* np.power(self.bp_kernels, 2))


        mlt = learning_rate/(np.sqrt(self.kernels_M.copy()) + 0.00001)
        mlt = np.multiply(mlt, self.bp_kernels)
        
        self.kernels = np.subtract(self.kernels, mlt)




