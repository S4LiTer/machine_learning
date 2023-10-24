import tkinter as tk
import threading
import numpy as np
# import matplotlib.pyplot as plt
import neural_network

class DrawingApp:
    def __init__(self, width=504, height=504, sample_height=28, sample_width=28):
        self.width = width
        self.height = height

        self.pic = np.zeros((sample_width, sample_height))

        self.sample_height = sample_height
        self.sample_width = sample_width

        self.pixel_size_x = width / sample_width
        self.pixel_size_y = width / sample_height

        self.running = False
        threading.Thread(target=self.start_window).start()

        while not self.running:
            continue

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

        self.running = True
        self.root.mainloop()

    def clear(self):
        # plt.imshow(self.pic)
        # plt.show()

        self.canvas.delete("all")
        self.pic = np.zeros((self.sample_width, self.sample_height))

    def draw(self, event):
        x = event.x
        y = event.y

        x_index = int((x / self.width)  * self.sample_width)
        y_index = int((y / self.height) * self.sample_height)

        self.set_pixel(x_index, y_index, 0.2, 0.01)

    def erase(self, event):
        x = event.x
        y = event.y

        x_index = int((x / self.width)  * self.sample_width)
        y_index = int((y / self.height) * self.sample_height)

        self.set_pixel(x_index, y_index, 0, erase = True)

    def askAI(self):
        to_guess = self.pic.copy()
        to_guess = to_guess.reshape(784)
        predict = self.AI.Calculate(to_guess)
        predict = predict.tolist()

        index = 0
        while index < 10:
            m = max(predict)
            m_index = predict.index(m)

            prob = round(predict[m_index] * 100, 2)
            predict[m_index] = 0

            self.texts[index].set(f"{self.nums[m_index]}: {prob}")

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

        if spread != None:
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

        self.askAI()

    def create_ranking(self):
        label_text = tk.StringVar()
        label = tk.Label(self.root, textvariable=label_text, font=('Arial', 30), bg="#252525", fg="#FFFFFF")
        label_text.set("AI predicts:")
        label.grid(row = 0, column = 4)

        self.labels = []
        self.texts = []
        self.nums = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")

        for i in range(10):
            self.texts.append(tk.StringVar())
            self.labels.append(tk.Label(self.root, textvariable=self.texts[i], font=('Arial', 20), bg="#252525", fg="#FFFFFF"))
            self.texts[i].set(f"{self.nums[i]}: 0%")
            self.labels[i].grid(row = i+1, column = 4, sticky="w", padx=30)

        self.AI = neural_network.NeuralNetwork(784, plot=False)
        self.AI.add_layer(256, "relu")
        self.AI.add_layer(64, "relu")
        self.AI.add_layer(10, "sigmoid")
        self.AI.storeNetwork(2, "load")




d = DrawingApp()