import numpy as np
import random
import time
import math

class Perceptron:
    def __init__(self, number_of_inputs) -> None:
        self.input_count = number_of_inputs

        weights = [random.uniform(-0.5, 0.5) for _ in range(number_of_inputs)]
        
        self.bias = 0
        self.weights = np.array(weights)

    def load(self, file):
        self.weights = np.loadtxt(file, delimiter=',')

    def save(self, file):
        np.savetxt(file, self.weights, delimiter=',')

    def calculate(self, input) -> float:
        multiply = np.multiply(input, self.weights)
        
        sum = np.sum(multiply) + self.bias
            
        self.prob = self.activation(sum)
        return self.prob
    
    def learn_brute_force(self, samples, labels):


        for i in range(len(self.weights)):
            print(f"{i}/{len(self.weights)}")
            start_time = time.time()
            self.gradient_descent_brute_force(i, samples, labels)
            print("finish:", round(time.time() - start_time, 5), "s")

        self.bias = self.calculate_error(samples, labels)
    
    def gradient_descent_brute_force(self, weight_index, samples, labels, speed = 50):
        diff = 0.0001
        tolerance = 0.001
        x1 = self.calculate_error(samples, labels)
        self.weights[weight_index] += diff
        x2 = self.calculate_error(samples, labels)
        
        derivation = x2-x1
        derivation = derivation/diff

        while abs(derivation) > tolerance:
            self.weights[weight_index] -= derivation*speed
            x1 = self.calculate_error(samples, labels)
            self.weights[weight_index] += diff
            x2 = self.calculate_error(samples, labels)
            
            derivation = x2-x1
            
            derivation = derivation/diff



    def learn_backpropagation(self, samples, labels, speed = 0.9):
        bp_weights, bp_bias = self.calculate_backpropagation(samples, labels)
        sm = np.sum(bp_weights)

        while abs(sm) > 0.075:
            bp_weights, bp_bias = self.calculate_backpropagation(samples, labels)
            self.weights = np.subtract(self.weights, bp_weights*speed)
            sm = np.sum(bp_weights)
            print("sum of backpropagation:", sm)


        while abs(bp_bias) > 0.00005:
            bp_weights, bp_bias = self.calculate_backpropagation(samples, labels, bias_only=True)
            self.bias -= bp_bias*speed
            print("backpropagation of bias:", bp_bias)
    

        return bp_weights, bp_bias

    def calculate_backpropagation(self, samples, labels, bias_only = False):
        index = 0
        backprop_weights = [0.0 for _ in range(self.input_count)]
        backprop_weights = np.array(backprop_weights)
        backprop_bias = 0

        for sample in samples:
            label = labels[index]

            multiply = np.multiply(sample, self.weights)
            z = np.sum(multiply) + self.bias

            output = self.activation(z)


            backprop_bias += self.d_activation(z)*2*(output-label*5)
            index += 1

            if bias_only:
                continue

            for i in range(self.input_count):            
                backprop_weights[i] += sample[i]*self.d_activation(z)*2*(output - label*5)
            

        return backprop_weights/len(samples), backprop_bias/len(samples)



    def activation(self, x):
        return 1/(1+pow(math.e, -x))

    def d_activation(self, x):
        return self.activation(x)*(1-self.activation(x))

    def calculate_error(self,samples, labels):
        index = 0
        error = 0
        
        for sample in samples:
            predict = self.calculate(sample)
            error += (predict - labels[index]*4)**2
            index += 1
        

        return error/len(labels)

    def test_network(self, samples, labels):
        i = 0
        correct_1 = 0
        total_1 = 0

        correct_0 = 0
        total_0 = 0

        while i < len(samples):
            pred = self.calculate(samples[i])
            if labels[i] == 1:
                total_1 += 1
                if round(pred) == 1:
                    correct_1 += 1

            if labels[i] == 0:
                total_0 += 1
                if round(pred) == 0:
                    correct_0 += 1

            i += 1

        print(f"Correctly fired: {correct_1}/{total_1}")
        print(f"Correctly not fired: {correct_0}/{total_0}")
        print(f"Total accuracy: {int(round((correct_1+correct_0)/(total_0+total_1), 2)*100)}%")

    def display_backpropagation(self, bp):
        index = 0
        for v in bp:
            char = "."
            if v > 0.1:
                char = "@"
            elif v > 0.075:
                char = "#"
            elif v > 0.05:
                char = "&"
            elif v > 0.04:
                char = "="
            elif v > 0.25:
                char = "+"
            elif v > 0.02:
                char = "-"
            elif v > 0.01:
                char = ":"
            elif v < 0:
                char = "O"

            print(char, end=" ")

            if not index%28:
                print()
            index += 1

        print()

