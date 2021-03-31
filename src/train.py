import os

import data
from tool.animation_imgs import PlotXY
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

# 대략 1.5만개 데이터 한번 학습하는데 걸린 시간
# macbook pro 2020(m1 16G) 11~16s (ㅗㅜㅑ...)
# macbook pro 2020(i5 16G) 77 ~ 90s
# colab 5s
def train_only(net, data_set):
    net.fit(data_set[0], data_set[1], epochs=65, batch_size=100,
            validation_data=(data_set[2], data_set[3]))

def load_data_set_and_train():
    data_set = load_data_set()
    PlotXY(data_set[0], data_set[1])

    # 전처리에 쓰레드풀 사용하여 tensorflow gpu 설정은 그 이후에 해야 함.
    from tensorflow.python.compiler.mlcompute import mlcompute
    mlcompute.set_mlc_device(device_name='gpu')
    import network

    net = network.load_v3()

    train_only(net, data_set)

    return (net, data_set)

    