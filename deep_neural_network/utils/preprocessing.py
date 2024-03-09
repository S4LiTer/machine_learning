import numpy as np
import random
import math
import time


def move_random(matrix):
    summed_matrix = np.sum(matrix, 0)


    low = 0
    index = 0
    while summed_matrix.sum(0)[index] == 0:
        low -= 1
        index += 1

    top = 0
    index = len(summed_matrix.sum(0))-1
    while index >= 0 and summed_matrix.sum(0)[index] == 0:
        top += 1
        index -= 1
    
    if top > 0:
        x_axis = random.randrange(low, top, 1)
        matrix = np.roll(matrix, x_axis, axis=2)


    low = 0
    index = 0
    while summed_matrix.sum(1)[index] == 0:
        low -= 1
        index += 1

    top = 0
    index = len(summed_matrix.sum(1))-1
    while index >= 0 and summed_matrix.sum(1)[index] == 0:
        top += 1
        index -= 1

    if top > 1:
        y_axis = random.randrange(low, top, 1)

        matrix = np.roll(matrix, y_axis, axis=1)

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

def scale_random(matrix, min_scale=0.65):
    scale = random.uniform(1, min_scale)

    scaled_matrix = np.zeros(matrix.shape)
    middle_x = ( matrix.shape[-1]-1 ) / 2
    middle_y = ( matrix.shape[-2]-1 ) / 2

    for y in range(matrix.shape[-2]):
        new_y = round((y - middle_y) * scale + middle_y)
        
        for x in range(matrix.shape[-1]):
            new_x = round((x - middle_x) * scale + middle_x)

            pixel = matrix[0, y, x]
            scaled_matrix[0, new_y, new_x] = pixel

    return scaled_matrix

def preprocess_array(default_array):
    output = np.empty_like(default_array)

    print(f"Preprocessing {output.shape[0]} images.")
    start_time = time.time()

    for index, pic in enumerate(default_array):

        pic = scale_random(pic)
        pic = move_random(pic)
        #pic = add_noise(pic)

        if not index%1000:
            print("Preprocessed so far:", index)

        output[index] = pic

    print("Preprocessing finished")
    print("total time:", round(time.time() - start_time, 2), "s")

    return output 


def add_noise(matrix, max_value=0.25, max_count=80):
    channel = random.randint(0, matrix.shape[0]-1)
    column = random.randint(0, matrix.shape[1]-1)
    row = random.randint(0, matrix.shape[2]-1)

    count = random.randint(0, max_count)

    for i in range(count):
        channel = random.randint(0, matrix.shape[0]-1)
        column = random.randint(0, matrix.shape[1]-1)
        row = random.randint(0, matrix.shape[2]-1)

        noise = random.random() * max_value

        matrix[channel, column, row] += noise

    return matrix


"""

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

"""