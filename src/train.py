import os, sys
import gc

sys.path.append("./src")

import data
import network
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    data_model = data.DataModel("./data/x", "./data/y", (256, 256, 3), (256, 256, 3), (50, 50, 2))
    result = data.load(data_model)
    # 13320 * 256
    x_train = np.array(result[0], dtype=data_model.dtype) / 255
    y_train = (np.array(result[1], dtype=data_model.dtype) / 255)
    x_eval = np.array(result[2], dtype=data_model.dtype) / 255
    y_eval = (np.array(result[3], dtype=data_model.dtype) / 255)
    x_test = np.array(result[4], dtype=data_model.dtype) / 255
    y_test = (np.array(result[5], dtype=data_model.dtype) / 255)

    net = network.load()

    no_trained_y = net.predict(x_test)

    print(f"gc ----- {gc.collect()}")


    net.fit(x_train, y_train, epochs=3, batch_size=100, validation_data=(x_eval, y_eval))

    trained_y = net.predict(x_test)

    real_y = y_test

    no_trained_y = np.insert(no_trained_y, 2, 0, -1)

    trained_y = np.insert(trained_y, 2, 0, -1)

    real_y = np.insert(real_y, 2, 0, -1)

    for i in range(real_y.shape[0]):
        plt.imshow(no_trained_y[i], vmin=0., vmax=1.)
        plt.show()
        plt.imshow(trained_y[i], vmin=0., vmax=1.)
        plt.show()
        plt.imshow(real_y[i], vmin=0., vmax=1.)
        plt.show()

