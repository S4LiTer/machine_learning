import neural_network
import layers
import matplotlib.pyplot as plt
from utils import activation_functions
import numpy as np

pool = layers.PoolingLayer((2, 4, 4), (2, 2))

tt = np.random.random((2, 4, 4))
res = pool.forward_pass(tt, True)
tt = np.random.random((2, 4, 4))
res = pool.forward_pass(tt, True)

grad_lsit = np.round((np.random.random((2, 2, 2, 2))*8)+1)
print(grad_lsit)
print(pool.indicies)
res2 = pool.backward_pass(grad_lsit, learning_rate=0)
print(res2)
"""
input_grad = np.zeros((9, 4))

inds = pool.indicies[0]
inds = inds.reshape((-1))

gradient = np.round(np.random.random((1, 9, 9))*3)
print(gradient)
gradient = gradient.reshape((1, 3, 3, 3, 3))
gradient = np.rot90(gradient, k=1, axes=(3, 2))[:, :, :, ::-1, :]
gradient = gradient.reshape((9, 9))

for index, grad in enumerate(gradient):
    input_grad[index, inds[index]] = grad[inds[index]]


input_grad = input_grad.reshape(1, 3, 3, 3, 3)[:, :, :, ::-1, :]
input_grad = np.rot90(input_grad, k=-1, axes=(3, 2))
input_grad = input_grad.reshape((1, 9, 9))
print(input_grad)
#print(gradient)
#print(gradient.shape)"""