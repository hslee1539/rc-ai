import os

import data
import os
import sys
import gc


def load_data_set():
    data_model = data.DataModel(
        "../data/x", "../data/y", (100, 100, 3), (100, 100, 3), (25, 25, 2))
    data_model.input_shape = (125, 125, 3)
    data_model.output_shape = (28, 28, 2)
    data_model.pre_shape = (125, 125, 3)
    return data.load(data_model)

# m1 mac 11~16s (우와...) CPU 4% GPU 80% * totalUse
# colab 5s
def train_only(net, data_set):
    net.fit(data_set[0], data_set[1], epochs=65, batch_size=100,
            validation_data=(data_set[2], data_set[3]))
    