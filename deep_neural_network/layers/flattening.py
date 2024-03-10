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
        # Tato vrstva nemá žádné hodnoty pro uložení, ale funkce tady musí být, aby bylo možné iterovat před všechny vrstvy a volat tuto funkci u každé
        
        return
    

    def forward_pass(self, input_matrix: np.ndarray, save_inputs=False) -> np.ndarray:
        # Převede tvar to jednozorměrného vektoru.
        # Pokud input_matrix zahrnuje více vzorků najednou (neboli jestli to je čtyřrozměrná matice), 
        # převede data každého vzorku do vektoru zvlášt a výstupen bude dvojrozměrná matice

        if len(input_matrix.shape) == len(self.input_size):
            return input_matrix.reshape((self.output_size))
        
        elif len(input_matrix.shape) == len(self.input_size) + 1:
            return input_matrix.reshape((input_matrix.shape[0], self.output_size))
        
        else:
            print("[ERROR] Invalid input to flattening layer")
    
    def backward_pass(self, output_gradient_list, learning_rate=0):
        # Převede data pro každý vzorek do stejného tvaru, jako tvar vstupu do této vrstvy
        return output_gradient_list.reshape((output_gradient_list.shape[0],) + self.input_size)
        