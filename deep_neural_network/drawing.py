from utils import preprocessing
import neural_network
import tkinter as tk
import numpy as np
import threading
import random

try:
    from mnist import MNIST
    mnist_installed = True
except:
    mnist_installed = False

try:
    import emnist
    mnist_installed = True
except:
    mnist_installed = False

class DrawingApp:
    def __init__(self, width=504, height=504, sample_height=28, sample_width=28, charmap_path=None, testing_samples=None):
        self.charmap_path = charmap_path

        self.width = width
        self.height = height

        self.pic = np.zeros((sample_width, sample_height))
        self.sample_height = sample_height
        self.sample_width = sample_width

        self.pixel_size_x = width / sample_width
        self.pixel_size_y = width / sample_height

        self.selected_label = None

        self.running = False
        threading.Thread(target=self.start_window).start()


        while not self.running:
            continue

        if testing_samples:
            self.show_random_btn["state"] = tk.ACTIVE
            self.test_network(testing_samples)
        


    def start_window(self):
        self.root = tk.Tk()
        self.root.geometry(f"{self.height+250}x{self.width+50}")
        self.root.title("Learning progression")
        self.root.configure(background="#252525")

        self.canvas = tk.Canvas(self.root, height=self.height, width=self.width, bg="#000000", highlightthickness=0)
        self.canvas.bind("<Button-1>", self.draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-3>", self.erase)
        self.canvas.bind("<B3-Motion>", self.erase)
        self.canvas.grid(row = 0, column = 0, rowspan= 20, columnspan=3)

        self.reset_btn = tk.Button(self.root, text="Clear", command = self.clear)
        self.reset_btn.grid(row=21, column=0, pady = 10)

        self.create_ranking()

        self.show_random_btn = tk.Button(self.root, text="Show random sample", command = self.show_random_sample, state=tk.DISABLED)
        self.show_random_btn.grid(row=21, column=1, pady=10)

        self.show_wrong_btn = tk.Button(self.root, text="Show random wrong guess", command = self.show_wrong_guess, state=tk.DISABLED)
        self.show_wrong_btn.grid(row=21, column=2, pady=10)


        self.running = True
        self.root.mainloop()

    def clear(self):
        # plt.imshow(self.pic)
        # plt.show()

        self.selected_label = None
        self.canvas.delete("all")
        self.pic = np.zeros((self.sample_width, self.sample_height))

    def draw(self, event):
        x = event.x
        y = event.y

        x_index = int((x / self.width)  * self.sample_width)
        y_index = int((y / self.height) * self.sample_height)

        self.selected_label = None

        self.set_pixel(x_index, y_index, 0.3, 0.02)
        self.askAI()

    def erase(self, event):
        x = event.x
        y = event.y

        x_index = int((x / self.width)  * self.sample_width)
        y_index = int((y / self.height) * self.sample_height)

        self.set_pixel(x_index, y_index, 0, erase = True)

    def askAI(self):
        to_guess = self.pic.copy()
        to_guess = to_guess.reshape(self.AI.network_input)
        predict = self.AI.Calculate(to_guess)

        index = 0
        
        while index < 10:
            m_index = np.argmax(predict)

            if self.selected_label == None:
                self.labels[index].config(fg="#FFFFFF")
            elif self.selected_label == m_index:
                self.labels[index].config(fg="#00FF00")
            elif self.selected_label != m_index:
                self.labels[index].config(fg="#FF0000")

            prob = round(predict[m_index] * 100, 2)
            predict[m_index] = 0

            self.texts[index].set(f"{self.characters[m_index]}: {prob}")

            index += 1

    def set_pixel(self, x_index, y_index, color, spread = None, erase = False):

        if y_index > 27 or x_index > 27:
            return
        
        x_start = round(self.pixel_size_x * x_index)
        x_end = round(self.pixel_size_x * (x_index + 1))

        y_start = round(self.pixel_size_y * y_index)
        y_end = round(self.pixel_size_y * (y_index + 1))


        current_color = self.pic[y_index][x_index]
        color += current_color
        if color > 1:
            color = 1
        elif erase:
            color = 0
        self.pic[y_index][x_index] = color


        hex_color = "#" + hex(round(color*255))[2:]*3
        self.canvas.create_rectangle(x_start, y_start, x_end, y_end, fill=hex_color, outline="")

        if spread == None:
            return

        for _spr_x in range(3):
            spr_x = _spr_x - 1

            for _spr_y in range(3):
                spr_y = _spr_y - 1

                if spr_x == 0 and spr_y == 0:
                    continue

                near_y = y_index+spr_y
                near_x = x_index+spr_x

                if near_y > 27 or near_y < 0 or near_x > 27 or near_x < 0:
                    continue

                current_color = self.pic[near_y][near_x]
                real_spread = spread/( abs(spr_x)+abs(spr_y) )

                real_color = current_color + real_spread

                if real_color > 1:
                    real_color = 1
                
                self.pic[near_y][near_x] = real_color

                hex_spread = "#" + f"{round(real_color*255):02x}"*3


                x_start = round(self.pixel_size_x * near_x)
                x_end = round(self.pixel_size_x * (near_x + 1))
                y_start = round(self.pixel_size_y * near_y)
                y_end = round(self.pixel_size_y * (near_y + 1))


                self.canvas.create_rectangle(x_start, y_start, x_end, y_end, fill=hex_spread, outline="")


    def create_ranking(self, ai_index = 11):
        self.characters = [i for i in range(10)]
        if charmap_path:
            charmap = open(charmap_path, "r")
            lines = charmap.read().split('\n')[:-1]
            self.characters = [chr(int(line.split(" ")[1])) for line in lines]


        label_text = tk.StringVar()
        label = tk.Label(self.root, textvariable=label_text, font=('Arial', 30), bg="#252525", fg="#FFFFFF")
        label_text.set("AI predicts:")
        label.grid(row = 0, column = 4, padx=20)

        self.labels = []
        self.texts = []
        for i in range(10):
            self.texts.append(tk.StringVar())
            self.labels.append(tk.Label(self.root, textvariable=self.texts[i], font=('Arial', 20), bg="#252525", fg="#FFFFFF"))
            self.texts[i].set(f"{self.characters[i]}: 0%")
            self.labels[i].grid(row = i+1, column = 4, sticky="w", padx=30)


        self.AI = neural_network.NeuralNetwork(0, plot=False)
        self.AI.storeNetwork(ai_index, "load")

        self.input_shape = self.AI.network_input
        if type(self.AI.network_input) == int:
            self.input_shape = (self.AI.network_input, )
            

    def test_network(self, samples):
        self.testing_images = samples[0].reshape((samples[0].shape[0], ) + self.input_shape)
        self.testing_labels = samples[1]

        testing_labels = []
        for label in self.testing_labels:
            exp = np.array([0. for _ in range(10)])
            exp[label] = 1.
            testing_labels.append(exp)
        self.testing_labels = np.array(testing_labels)

        self.failed_samples, self.failed_labels = neural_network.Test(self.AI, self.testing_images, self.testing_labels, self.charmap_path)
        self.show_wrong_btn["state"] = tk.ACTIVE


    def show_random_sample(self):
        self.clear()
        index = random.randint(0, len(self.testing_images)-1)

        sample = self.testing_images[index].reshape(28, 28)
        self.selected_label = np.argmax(self.testing_labels[index])

        self.show_matrix(sample)

    def show_wrong_guess(self):
        self.clear()
        index = random.randint(0, len(self.failed_labels)-1)

        sample = self.failed_samples[index].reshape(28, 28)
        self.selected_label = self.failed_labels[index]
        
        self.show_matrix(sample)

    def show_matrix(self, matrix):
        for y in range(matrix.shape[1]):
            for x in range(matrix.shape[0]):
                self.set_pixel(x, y, matrix[y][x])

        self.askAI()



mndata = MNIST("samples")
_testing_images, testing_labels = mndata.load_testing()
testing_images = np.array(_testing_images)/255

testing_samples = (testing_images, testing_labels)

charmap_path = "samples/EMNIST/emnist-balanced-mapping.txt"
d = DrawingApp(charmap_path=charmap_path, testing_samples=testing_samples)