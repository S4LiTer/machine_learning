import math
import numpy as np


functions  = {
    "sigmoid": {
        "name"      : "sigmoid",
        "function"  : lambda x: sigmoid(x), 
        "derivation": lambda x: sigmoid_d(x)},
    "relu": {
        "name"      : "relu",
        "function"  : lambda x: relu(x),
        "derivation": lambda x: relu_d(x)},
    "softmax": {
        "name"      : "softmax",
        "function"  : lambda x: softmax(x),
        "derivation": None
    }
}


def sigmoid(x):
    return 1/(1+pow(math.e,-x))

def sigmoid_d(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    res = np.maximum(x,0)
    #print(np.sum(res))
    return res

def relu_d(x):
    x_copy = x.copy()
    x_copy[x_copy > 0] = 1
    x_copy[x_copy < 0] = 0
    return x_copy


def softmax(x):
    e_x = np.exp(x.T - np.max(x, axis=-1))
    sum = np.sum(e_x, axis=0).reshape(-1, 1)
    return  e_x.T / sum

