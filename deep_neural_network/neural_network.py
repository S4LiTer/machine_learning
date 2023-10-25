from utils import activation_functions as act_funcs
from utils import display
import preprocessing
import numpy as np
import layer
import time

class NeuralNetwork:
    def __init__(self, input_size, plot=True):
        self.plot = plot
        self.layers = []
        self.layer_sizes = [input_size]

        if self.plot:
            self.graph = display.Window(1000, 450, 500)

    def add_layer(self, output_size, activation_func):
        activation = act_funcs.functions[activation_func]

        new_layer = layer.Layer(self.layer_sizes[-1], output_size, activation)
    

        self.layers.append(new_layer)
        self.layer_sizes.append(output_size)

    def Calculate(self, prev_layer, save_inputs = False):
        for layer in self.layers:
            prev_layer = layer.predict(prev_layer, save_inputs)

        return prev_layer
    

    def Backpropagate(self, last_output, expected_output, learning_rate: float):
        gradient = 2*np.subtract(last_output, expected_output)


        i = len(self.layers)-1
        while i >= 0:
            gradient = self.layers[i].adjust(gradient, learning_rate)
            i -= 1


    def storeNetwork(self, id, action="save", path="saved_networks/"):
        order = 0
        for layer in self.layers:
            layer.storeValues(order, id, action, path)
            order += 1


    def Train(self, samples, labels, testing_samples, testing_labels, batch_size: int, learning_rate, gens):
        samples_count = len(samples)

        for gen in range(gens):
            """
            if gen%4 == 0:
                noised_samples = samples.copy()
                for sample in noised_samples:
                    preprocessing.add_noise(sample)
            """

            i = 0

            gen_start_time = time.time()

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


            gen_total_time = time.time() - gen_start_time
            testing_accouracy, testing_loss = self.GetAccouracy(testing_samples[:2000], testing_labels[:2000])
            accouracy, loss = self.GetAccouracy(samples[:2000], labels[:2000])

            print("gen:", str(gen) + ", time to calculate: ", round(gen_total_time, 1), "s")
            print("accouracy:", str(round(accouracy, 2))+"%, loss:", str(round(loss, 0)))
            print("Testing accouracy:", str(round(testing_accouracy, 2))+"%, testing loss:", str(round(testing_loss, 0)))
            
            if self.plot:
                self.graph.add_point(accouracy, 0)
                self.graph.add_point(testing_accouracy, 1)




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

        total_loss = 0

        for sample in samples:
            label = labels[label_index]
            label_index += 1

            res = self.Calculate(sample)
            correct_index = np.where(label == max(label))[0].astype(int)[0]
            guess_index = np.where(res == max(res))[0].astype(int)[0]

            cost = np.subtract(res, label)**2
            total_loss += np.sum(cost, axis=0)


            if guess_index == correct_index:
                total_correct += 1

        total_accouracy = round(total_correct/len(labels), 4)*100
        return total_accouracy, total_loss
        #print(f"accouracy: {total_accouracy}, time: {time.time() - start_time}")


    def Test(self, samples, labels):
        label_index = 0

        total_raw = len(labels)
        total_correct_raw = 0

        failed_samples = []
        failed_labels = []

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
            else:
                failed_samples.append(sample)
                failed_labels.append(correct_index)
        
        ind = 0
        for n in total:
            tc = total_correct[ind]
            ind += 1

            print(f"Number {ind}: {tc}/{n} ({round((tc/n)*100)}%)")
        

        print(f"Total: {total_correct_raw}/{total_raw} ({round((total_correct_raw/total_raw)*100)}%)")

        return failed_samples, failed_labels
