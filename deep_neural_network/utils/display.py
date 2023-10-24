import tkinter as tk
import threading
import time


class Window:
    def __init__(self, width, height, values_per_graph = 100):
        self.width = width
        self.height = height
        self.values_per_graph = values_per_graph
        self.line_color = ["#8d34eb", "#ee4b2b"]
        self.grid_color = "#656565"
        self.axis_color = "#9C9C9C"
        self.grid_offset = 20

        self.data_count = [0 for _ in range(len(self.line_color))]
        self.x_step = width/values_per_graph
        self.y_step = (self.height-40)/10
        self.last_y_value = [self.height-self.grid_offset for _ in range(len(self.line_color))]

        self.running = False
        threading.Thread(target=self.start_window).start()

        while not self.running:
            continue


    def start_window(self):
        self.root = tk.Tk()
        self.root.title("Learning progression")

        self.canvas = tk.Canvas(self.root, height=self.height, width=self.width, bg="#252525")
        self.canvas.pack()
        self.draw_grid()
        self.running = True
        self.root.mainloop()

    def draw_grid(self):
        for i in range(self.values_per_graph):
            line_x = (i+1)*self.x_step+self.grid_offset
            self.canvas.create_line(line_x, self.height, line_x, 0, fill="#3A3A3A", width=1) 

        
        
        for i in range(10):
            line_height = i*self.y_step+self.grid_offset
            self.canvas.create_line(0, line_height, self.width, line_height, fill=self.grid_color, width=1)


        self.canvas.create_line(0, self.height-self.grid_offset, self.width, self.height-self.grid_offset, fill=self.axis_color, width=2)
        self.canvas.create_line(self.grid_offset, 0, self.grid_offset, self.height, fill=self.axis_color, width=2)


    def add_point(self, y_value, index):
        if self.data_count[index] >= self.values_per_graph:
            self.reset()
            

        x_start = self.grid_offset + self.data_count[index]*self.x_step
        self.data_count[index] += 1

        current_y_value = self.height-self.grid_offset-((y_value/10)*self.y_step)


        self.canvas.create_line(x_start, self.last_y_value[index], x_start+self.x_step, current_y_value, fill=self.line_color[index], width=2)

        self.last_y_value[index] = current_y_value

    def reset(self):
        self.data_count = 0
        self.canvas.delete("all")
        self.draw_grid()


