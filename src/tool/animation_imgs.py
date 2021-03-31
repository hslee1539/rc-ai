import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Button
import numpy as np

class PlotXY:
    def __init__(self, xs: np.array, ys: np.array):
        self.xs = xs
        self.ys = ys
        fig = plt.figure()
        fig : plt.Figure

        self.x_axes = fig.add_subplot("131")
        self.y1_axes = fig.add_subplot("132")
        self.y2_axes = fig.add_subplot("132")
        self.x_axes : Axes
        self.y1_axes : Axes
        self.y2_axes : Axes

        self.index = 0
        self.index_max = xs.shape[0]

        self.x_img = self.x_axes.imshow(self.xs[self.index % self.index_max])
        self.y1_img = self.y1_axes.imshow(self.ys[self.index % self.index_max, :, :, 0])
        self.y2_img = self.y2_axes.imshow(self.ys[self.index % self.index_max, :, :, 1])

        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.nextB)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(self.previous)

        plt.show()

    def nextB(self, event):
        self.index += 1
        self.x_img.set_data(self.xs[self.index % self.index_max])
        self.y1_img.set_data(self.ys[self.index % self.index_max, :, :, 0])
        self.y2_img.set_data(self.ys[self.index % self.index_max, :, :, 0])

        plt.draw()
    
    def previous(self, event):
        self.index -= 1
        self.x_img.set_data( self.xs[self.index % self.index_max])
        self.y1_img.set_data(self.ys[self.index % self.index_max, :, :, 0])
        self.y2_img.set_data(self.ys[self.index % self.index_max, :, :, 0])

        plt.draw()



if __name__ == "__main__":
    x = np.random.rand(10, 100, 100, 3)
    y = np.random.rand(10, 50, 50, 2)

    PlotXY(x,y)


    


    
    
    