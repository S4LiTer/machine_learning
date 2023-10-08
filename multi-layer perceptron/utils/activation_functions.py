import math
import numpy as np


functions  = {
    "sigmoid": {"function"  : lambda x: sigmoid(x), 
                "derivation": lambda x: sigmoid_d(x)},
    "relu": {
        "function"  : lambda x: relu(x),
        "derivation": lambda x: relu_d(x)}
}


def sigmoid(x):
    return 1/(1+pow(math.e,-x))

def sigmoid_d(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return np.maximum(x,0)

def relu_d(x):
    x_copy = x.copy()
    x_copy[x_copy > 0] = 1
    x_copy[x_copy < 0] = 0
    return x_copy


"""
z nejakeho duvodu je vsechno 0. to znamena, ze derivace relu je taky nula a sit se nic neuci.
Je to divne, protoze zadna hodnota ve vahach nebo biasech neni defaultne nastavena zaporne, takze by se to tak nemelo dostat
(obvzlaste ne takhle rychle)
"""