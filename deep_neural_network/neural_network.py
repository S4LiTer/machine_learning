from utils import activation_functions as act_funcs
import matplotlib.pyplot as plt
from utils import preprocessing
from utils import display
import numpy as np
import layers
import time
import json

class NeuralNetwork:
    def __init__(self, input_size, plot=True):
        self.plot = plot

        self.layers = []
        self.network_input = input_size

        if self.plot:
            self.graph = display.Window(1000, 450, 50)


    def add_fully_connected_layer(self, output_size: int, activation_func="relu", optimizer="RMSprop"):
        activation = act_funcs.functions[activation_func]
        
        input_size = self.network_input
        if len(self.layers) > 0:
            input_size = self.layers[-1].output_size

        new_layer = layers.FullyConnectedLayer(input_size, output_size, activation, optimizer)
    

        self.layers.append(new_layer)

    def add_flattening_layer(self):

        input_size = self.network_input
        if len(self.layers) > 0:
            input_size = self.layers[-1].output_size

        new_layer = layers.FlatteningLayer(input_size)

        self.layers.append(new_layer)

    def add_pooling_layer(self, pool_size: tuple, pooling_type=layers.pooling_types.max_pooling):
        input_size = self.network_input
        if len(self.layers) > 0:
            input_size = self.layers[-1].output_size

        new_layer = layers.PoolingLayer(input_size, pool_size, pooling_type)

        self.layers.append(new_layer)

    def add_convolutional_layer(self, kernel_size:tuple, kernel_count:int, correlation_type = layers.conv_types.valid, activation_func = "relu", optimizer="RMSprop"):
        activation = act_funcs.functions[activation_func]
        
        input_size = self.network_input
        if len(self.layers) > 0:
            input_size = self.layers[-1].output_size

        new_layer = layers.ConvolutionalLayer(input_size, kernel_size, kernel_count, correlation_type, activation, optimizer)

        self.layers.append(new_layer)


    def Calculate(self, prev_layer, save_inputs = False):
        for layer in self.layers:
            prev_layer = layer.forward_pass(prev_layer, save_inputs)

        return prev_layer



    def storeNetwork(self, id: int, action="save", path="saved_networks/"):
        filename = f"{path}{id}__network_config"

        if action == "save":
            data = {"network_input": self.network_input, "layers": []}
            
            index = 0
            for order, layer in enumerate(self.layers):
                layer.storeValues(order, id, action, path)
                data["layers"].append(layer.layer_data)
                index += 1
                
            with open(filename, 'w') as file:
                json.dump(data, file, indent=2)

        
        else:
            self.layers = []

            conf = open(filename, "r")
            data = json.load(conf) 

            self.network_input = data["network_input"]
            if type(self.network_input) == list:
                self.network_input = tuple(self.network_input)

            for order, layer in enumerate(data["layers"]):
                self.load_layer(layer)

                self.layers[-1].storeValues(order, id, action, path)

            conf.close()

        order = 0
        for layer in self.layers:
            layer.storeValues(order, id, action, path)
            order += 1
            

    def load_layer(self, layer_data: dict):
        layer_type = layer_data["layer_type"]
        
        if layer_type == "fully_connected":
            output = layer_data["output_size"]
            activation = layer_data["activation"]
            optimizer = layer_data["optimizer"]
            self.add_fully_connected_layer(output, activation, optimizer)

        elif layer_type == "flattening":
            self.add_flattening_layer()

        elif layer_type == "pooling":
            pool_size = tuple(layer_data["pool_size"])
            pooling_type = layer_data["pooling_type"]
            self.add_pooling_layer(pool_size, pooling_type)

        elif layer_type == "convolutional":
            kernel_size = tuple(layer_data["kernel_size"])
            kernel_count = layer_data["kernel_count"]
            correlation_type = layer_data["correlation_type"]
            activation = layer_data["activation"]
            optimizer = layer_data["optimizer"]

            self.add_convolutional_layer(kernel_size, kernel_count, correlation_type, activation, optimizer)

    def Backpropagate(self, last_output: np.ndarray, expected_output: np.ndarray, learning_rate: float):
        gradient = last_output - expected_output

        i = len(self.layers)-1
        while i >= 0:
            gradient = self.layers[i].backward_pass(gradient, learning_rate)
            i -= 1


    def Train(self, samples: np.ndarray, labels, testing_samples, testing_labels, batch_size: int, learning_rate: float, gens: int):
        samples_count = len(samples)

        for gen in range(gens):
            i = 0
            gen_start_time = time.time()

            permutation = np.random.permutation(samples_count)


            shuffled_samples = samples[permutation]
            shuffled_labels = labels[permutation]


            batch_number = 0
            batch_count = samples_count//batch_size
            debug_interval = batch_count//30
            batch_time = time.time()

            
            while i+batch_size < samples_count:
                batch_labels = shuffled_labels[i:i+batch_size]
                batch_samples = shuffled_samples[i:i+batch_size]
                
                # calculates predict for every item in batch (starts in i index and go through batch_size samples)
                # input values for each layer are stored and will be used in  backpropagation process
                
                predicts = self.Calculate(batch_samples, True)

                self.Backpropagate(predicts, batch_labels, learning_rate)
                i += batch_size
                batch_number += 1

                if not batch_number % debug_interval:
                    print("-"*64)
                    print(f"progress: {batch_number}/{batch_count}")
                    print(f"{debug_interval} batches completed in: {round(time.time()-batch_time, 2)}s (gen {gen})")
                    epoch_finish = round(((batch_count-batch_number)//debug_interval) * (time.time()-batch_time))
                    finish = (round((batch_count//debug_interval) * (time.time()-batch_time)) * (gens-gen-1)) + epoch_finish
                    print(f"estimated epoch finish in {epoch_finish} s")
                    print(f"estimated total finish in {finish} s")
                    accuracy = np.mean(np.equal(np.argmax(predicts, axis=1), np.argmax(batch_labels, axis=1))) * 100
                    print(f"batch accouracy: {round(accuracy, 2)}%")
                    
                    batch_time = time.time()



            gen_total_time = time.time() - gen_start_time
            print("gen:", str(gen) + ", time to calculate: ", round(gen_total_time, 1), "s")
            if self.plot:
                
                testing_accouracy = self.GetAccouracy(testing_samples[:4000], testing_labels[:4000])
                accouracy = self.GetAccouracy(samples[:4000], labels[:4000])

                print("accouracy:", str(round(accouracy, 2))+"% || Testing accouracy:", str(round(testing_accouracy, 2))+"%")
            
                self.graph.add_point(accouracy, 0)
                self.graph.add_point(testing_accouracy, 1)


    def GetAccouracy(self, samples, labels):
        results = self.Calculate(samples)
        results = np.argmax(results, axis=1)
        ind_labels = np.argmax(labels, axis=1)
        
        accouracy = np.sum(results == ind_labels)/len(results)

        return accouracy*100



def Test(nn, samples, labels, charmap_path = None):
        characters = None
        if charmap_path:
            charmap = open(charmap_path, "r")
            lines = charmap.read().split('\n')[:-1]
            characters = [chr(int(line.split(" ")[1])) for line in lines]


        predicts = nn.Calculate(samples)

        failed_samples = []
        failed_labels = []

        total_output_classes = np.zeros((labels.shape[1:]))
        correct_output_classes = np.zeros((labels.shape[1:]))


        for sample, label, predict in zip(samples, labels, predicts):
            total_output_classes[np.argmax(label)] += 1

            if np.argmax(label) == np.argmax(predict):
                correct_output_classes[np.argmax(label)] += 1
                continue
            
            failed_samples.append(sample)
            failed_labels.append(np.argmax(label))
            
        for index, output_class in enumerate(total_output_classes):
            char = index
            if characters:
                char = characters[index]
            print(f"{char}: {round((correct_output_classes[index]/output_class)*100, 1)}% ({int(correct_output_classes[index])}/{int(output_class)})")
        
        print(f"total accouracy: {round((np.sum(correct_output_classes)/np.sum(total_output_classes))*100, 1)}% ({int(np.sum(correct_output_classes))}/{int(np.sum(total_output_classes))})")
        return failed_samples, failed_labels