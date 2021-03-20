import os, sys

sys.path.append("./src")

import data
import network
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    data_model = data.DataModel("./data/x", "./data/y", (256, 256, 3), (256, 256, 3), (50, 50, 2))
    result = data.load(data_model)
    # 13320 * 256
    x_train = np.array(result[0]) / 255
    y_train = (np.array(result[1]) / 255)[:,:,:,0:2]
    x_eval = np.array(result[2]) / 255
    y_eval = (np.array(result[3]) / 255)[:,:,:,0:2]
    x_test = np.array(result[4]) / 255
    y_test = (np.array(result[5]) / 255)[:,:,:,0:2]

    net = network.load()

    shape = list(data_model.output_shape)
    shape[2] = 3
    no_trained_y = np.zeros(shape)
    no_trained_y[:,:,0:2] = net.predict(x_test[0:1])


    net.fit(x_train, y_train, epochs=20, batch_size=100, validation_data=(x_eval, y_eval))

    trained_y = np.zeros(shape)
    trained_y[:,:,0:2] = net.predict(x_test[0:1])

    real_y = np.zeros(shape)
    real_y[:,:,0:2] = y_test[0]

    plt.imshow(no_trained_y, vmin=0., vmax=1.)
    plt.show()
    plt.imshow(trained_y, vmin=0., vmax=1.)
    plt.show()
    plt.imshow(real_y)
    plt.show()

