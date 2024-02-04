import matplotlib.pyplot as plt
import numpy as np
import os



def show_1():
    input_matrix = np.loadtxt(f"{os.path.dirname(__file__)}/samples/1_0_input", delimiter=',')
    result = [np.loadtxt(f"{os.path.dirname(__file__)}/samples/1_1_result_{i}", delimiter=',') for i in range(3)]
    kernel = [np.loadtxt(f"{os.path.dirname(__file__)}/samples/1_2_kernel_{i}", delimiter=',') for i in range(3)]

    fig, axs = plt.subplots(2, 4)
    axs[0][0].imshow(input_matrix, cmap='gray')
    axs[0][0].set_title('INPUT')

    axs[0][1].imshow(result[0], cmap='gray')
    axs[0][1].set_title('RESULT 2')

    axs[0][2].imshow(result[1], cmap='gray')
    axs[0][2].set_title('RESULT 2')

    axs[0][3].imshow(result[2], cmap='gray')
    axs[0][3].set_title('RESULT 3')

    axs[1][0].axis("off")

    axs[1][1].imshow(kernel[0], cmap='gray')
    axs[1][1].set_title('RESULT 2')

    axs[1][2].imshow(kernel[1], cmap='gray')
    axs[1][2].set_title('RESULT 2')

    axs[1][3].imshow(kernel[2], cmap='gray')
    axs[1][3].set_title('RESULT 3')

    plt.tight_layout()

    plt.show()

show_1()