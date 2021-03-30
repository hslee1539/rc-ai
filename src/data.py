# 모듈
import os
import numpy as np
import PIL.Image as pilimg
import matplotlib.pyplot as plt
import multiprocessing
import random

# 내부 모듈

from data_function import find_file_name_sets, open_pre_sets, create_rotated_set_args, create_rotated_set
from data_model import DataModel
from threaded_generator import ThreadedGenerator


def convert_x(xy):
    return xy[0]


def convert_y(xy):
    return xy[1]


def load(data_model: DataModel):
    file_sets = find_file_name_sets(data_model)
    pre_sets = open_pre_sets(data_model, file_sets)
    pre_set_args = create_rotated_set_args(data_model, pre_sets)

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    x_eval = []
    y_eval = []
    with multiprocessing.Pool(8) as pool:
        rows = pool.map(create_rotated_set, pre_set_args)
        random.seed(100)
        random.shuffle(rows)

        index = 0
        for row in rows:
            if index < 100:
                x_eval.append(row[0] / 255 * 2 - 1)
                y_eval.append(row[1] / 255)
            elif (index // 100 * 100 / len(rows) < 0.7):
                x_train.append(row[0] / 255 * 2 - 1)
                y_train.append(row[1] / 255)
            else:
                x_test.append(row[0] / 255 * 2 - 1)
                y_test.append(row[1] / 255)
            index += 1

    return (
        np.array(x_train, dtype=data_model.dtype), 
        np.array(y_train, dtype=data_model.dtype), 
        np.array(x_eval, dtype=data_model.dtype), 
        np.array(y_eval, dtype=data_model.dtype), 
        np.array(x_test, dtype=data_model.dtype), 
        np.array(y_test, dtype=data_model.dtype))


if __name__ == "__main__":
    test_data_model = DataModel(
        "./data/x", "./data/y", (256, 256, 3), (256, 256, 3), (50, 50, 2))

    x, y, x1, y1, x2, y2 = load(test_data_model)

    for i in range(100):
        plt.imshow(x[i * 10])
        plt.show()
        plt.imshow(np.insert(y[i * 10], 2, 0, -1))
        plt.show()
