import data
import network
import matplotlib.pyplot as plt
import numpy as np


data_model = data.DataModel("./data/x", "./data/y", (256, 256, 3), (50, 50, 2))
x = data.load_img(data_model)
y = data.load_label_img(data_model)

net = network.load()

shape = list(data_model.output_shape)
shape[2] = 3
no_trained_y = np.zeros(shape)
no_trained_y[:,:,0:2] = net.predict(x[0:1])


net.fit(x, y, epochs=400)

trained_y = np.zeros(shape)
trained_y[:,:,0:2] = net.predict(x[0:1])

real_y = np.zeros(shape)
real_y[:,:,0:2] = y[0]

plt.imshow(no_trained_y, vmin=0., vmax=1.)
plt.show()
plt.imshow(trained_y, vmin=0., vmax=1.)
plt.show()
plt.imshow(real_y)
plt.show()


