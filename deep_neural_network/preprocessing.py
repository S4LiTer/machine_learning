from  mnist import MNIST # 28x28
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import random
import math
import time


def move_random(matrix):
    low = 0
    index = 0
    while matrix.sum(0)[index] == 0:
        low -= 1
        index += 1

    top = 0
    index = len(matrix.sum(0))-1
    while index >= 0 and matrix.sum(0)[index] == 0:
        top += 1
        index -= 1
    
    if top > 0:
        x_axis = random.randrange(low, top, 1)

        rotation = deque([i for i in range(matrix.shape[1])])
        rotation.rotate(x_axis)

        matrix = matrix[:, rotation]



    low = 0
    index = 0
    while matrix.sum(1)[index] == 0:
        low -= 1
        index += 1

    top = 0
    index = len(matrix.sum(1))-1
    while index >= 0 and matrix.sum(1)[index] == 0:
        top += 1
        index -= 1

    if top > 1:
        y_axis = random.randrange(low, top, 1)

        rotation = deque([i for i in range(matrix.shape[0])])
        rotation.rotate(y_axis)

        matrix = matrix[rotation, :]


    return matrix

def rotate_random(matrix):
    random_range = 0.3
    fi = random.uniform(-random_range, random_range)

    cos_fi = math.cos(fi)
    sin_fi = math.sin(fi)

    rotated_matrix = np.zeros(matrix.shape)

    for y in range(matrix.shape[0]):
        real_y = y - matrix.shape[0]/2 # real_x a real_y jsou x a y souřadnice, když je střed uprostřed

        for x in range(matrix.shape[1]):
            real_x = x - matrix.shape[1]/2

            pixel_value = matrix[y][x]

            if pixel_value != 0:
                new_real_x = real_x*cos_fi - real_y*sin_fi
                new_real_y = real_x*sin_fi + real_y*cos_fi

                new_x = round(new_real_x + matrix.shape[1]/2)
                new_y = round(new_real_y + matrix.shape[0]/2)

                if new_x >= rotated_matrix.shape[1] or new_y >= rotated_matrix.shape[0]:
                    continue

                rotated_matrix[new_y][new_x] = pixel_value


    for y in range(matrix.shape[0]-1):
        if y == 0: continue

        for x in range(matrix.shape[1]-1):
            if x == 0: continue

            pixel = rotated_matrix[y][x]
            if pixel > 0.05: continue
            

            nearby_pixels = rotated_matrix[y-1][x]
            nearby_pixels += rotated_matrix[y+1][x]
            nearby_pixels += rotated_matrix[y][x-1]
            nearby_pixels += rotated_matrix[y][x+1]

            if nearby_pixels > 2.1:
                rotated_matrix[y][x] = nearby_pixels/4


    

    return rotated_matrix

def scale_random(matrix):
    scale = random.uniform(1, 0.55)

    scaled_matrix = np.zeros(matrix.shape)
    middle_x = ( len(matrix[0])-1 ) / 2
    middle_y = ( len(matrix) -  1 ) / 2

    for y in range(matrix.shape[0]):
        new_y = round((y - middle_y) * scale + middle_y)
        
        for x in range(matrix.shape[1]):
            new_x = round((x - middle_x) * scale + middle_x)

            pixel = matrix[y][x]
            scaled_matrix[new_y][new_x] = pixel

    return scaled_matrix

def preprocess_array(default_array):
    array = default_array.copy()
    index = 0

    print(f"Preprocessing {len(array)} images.")
    start_time = time.time()

    for pic in array:
        sample = pic.reshape(28, 28)

        #sample = rotate_random(sample)
        sample = scale_random(sample)
        sample = move_random(sample)

        sample = sample.reshape(784)

        if not index%int(len(array)/10):
            print("Preprocessed so far:", index)

        array[index] = sample
        index += 1

    print("Preprocessing finished")
    print("total time:", round(time.time() - start_time, 2), "s")

    return array 


def add_noise(matrix):
    count = 50
    max_value = 0.2

    for i in range(count):
        pos = random.randrange(0, len(matrix))

        to_add = random.uniform(0, max_value)

        if matrix[pos] + to_add > 1:
            matrix[pos] = 1
            continue

        matrix[pos] += to_add

    return matrix




if __name__ == "__main__":
    start_time = time.time()
    mndata = MNIST("samples")
    _images, labels = mndata.load_training()
    images = np.array(_images)/255
    print("load:", time.time() - start_time, "s")


    preprocessed_array = preprocess_array(images)

    start_time = time.time()
    for sample in preprocessed_array:
        sample = add_noise(sample)

    print("Time took to add noise to all samples:", round(time.time() - start_time, 2), "s")


    for i in range(len(preprocessed_array)):
        preprocessed_sample = preprocessed_array[i].reshape(28, 28)
        sample = images[i].reshape(28, 28)
        print(i)

        f, axarr = plt.subplots(2)
        axarr[0].imshow(sample)
        axarr[1].imshow(preprocessed_sample)


        plt.show()