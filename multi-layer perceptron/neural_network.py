from utils import activation_functions as act_funcs
from utils import display
import numpy as np
import layer
import time

class NeuralNetwork:
    def __init__(self, neuron_count, activation_func="sigmoid"):
        activation = act_funcs.functions[activation_func]

        self.layers = [layer.Layer(neuron_count[i], neuron_count[i+1], activation) for i in range(len(neuron_count)-1)]


        self.graph = display.Window(1000, 450, 500)

    def Calculate(self, prev_layer, save_inputs = False):
        for layer in self.layers:
            prev_layer = layer.predict(prev_layer, save_inputs)

        return prev_layer
    

    def Backpropagate(self, last_output, expected_output, learning_rate: float):
        gradient = 2*np.subtract(last_output, expected_output)


        i = len(self.layers)-1
        while i >= 0:
            gradient = self.layers[i].adjust(gradient, learning_rate)
            # print(f"\nlayer {i}\n:", self.layers[i].bp_biases)
            i -= 1


    def storeNetwork(self, id, action="save", path="saved_networks/"):
        order = 0
        for layer in self.layers:
            layer.storeValues(order, id, action, path)
            order += 1


    def Train(self, samples, labels, batch_size: int, learning_rate, gens):
        samples_count = len(samples)

        for gen in range(gens):
            i = 0

            permutation = np.random.permutation(samples_count)
            shuffled_samples = samples[permutation]
            shuffled_labels = labels[permutation]
            

            while i+batch_size < samples_count:
                batch_labels = shuffled_labels[i:i+batch_size]

                # calculates predict for every item in batch (starts in i index and go through batch_size samples)
                # input values for each layer are stored and will be used in  backpropagation process
                predicts = np.array([self.Calculate(shuffled_samples[i+batch_index], True) for batch_index in range(batch_size)])

                self.Backpropagate(predicts, batch_labels, learning_rate)
                i += batch_size


            accouracy = self.GetAccouracy(samples, labels)
            self.graph.add_point(accouracy)
            print("gen:", gen, "accouracy:", str(accouracy)+"%")




    def CalculateLoss(self, samples, labels):
        total_loss = 0

        index = 0
        for sample in samples:
            label = labels[index]
            index += 1

            res = self.Calculate(sample)
            cost = np.subtract(res, label)**2
            total_loss += np.sum(cost, axis=0)

        return total_loss

    def GetAccouracy(self, samples, labels):
        start_time = time.time()
        label_index = 0
        total_correct = 0

        for sample in samples:
            label = labels[label_index]
            label_index += 1

            res = self.Calculate(sample)
            correct_index = np.where(label == max(label))[0].astype(int)[0]
            guess_index = np.where(res == max(res))[0].astype(int)[0]

            if guess_index == correct_index:
                total_correct += 1

        total_accouracy = round(total_correct/len(labels), 4)*100
        return total_accouracy
        #print(f"accouracy: {total_accouracy}, time: {time.time() - start_time}")


    def Test(self, samples, labels):
        label_index = 0

        total_raw = len(labels)
        total_correct_raw = 0

        total = [0 for _ in range(len(labels[0]))]
        total_correct = [0 for _ in range(len(labels[0]))]

        for sample in samples:
            label = labels[label_index]
            label_index += 1

            res = self.Calculate(sample)
            correct_index = np.where(label == max(label))[0].astype(int)[0]
            guess_index = np.where(res == max(res))[0].astype(int)[0]

            total[correct_index] += 1
            if guess_index == correct_index:
                total_correct_raw += 1
                total_correct[guess_index] += 1
        
        ind = 0
        for n in total:
            tc = total_correct[ind]
            ind += 1

            print(f"Number {ind}: {tc}/{n} ({round((tc/n)*100)}%)")
        

        print(f"Total: {total_correct_raw}/{total_raw} ({round((total_correct_raw/total_raw)*100)}%)")
