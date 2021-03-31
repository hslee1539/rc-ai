import tensorflow as tf
import numpy as np
import os
from data_model import DataModel

"""
1. 모델 생성(model.h5 가 없는 경우)
2. 모델 로딩(model.h5 가 있는 경우)
3. loadModel() : tensorflow.keras.model.Model 제공
4. saveModel(model : tensorflow.keras.model.Model) 제공
"""

SAVED_MODEL_PATH = os.path.dirname(
    __file__) + "/model"  # 경로 중 디렉토리명만 얻고 model 폴더 설정


# loss 0.069?
def load(path=SAVED_MODEL_PATH):
    model: tf.keras.Model

    try:
        model = tf.keras.models.load_model(path)
        print("load")
    except Exception as identifier:
        # 모델이 없는 경우
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(
            filters=5, kernel_size=(3, 3), strides=1, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(
            filters=7, kernel_size=(5, 5), strides=2, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(
            filters=13, kernel_size=(7, 7), strides=1, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(
            filters=17, kernel_size=(13, 13), strides=1, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Conv2D(
            filters=2, kernel_size=(5, 5), strides=1, activation="relu"))
        model.build((1, 100, 100, 3))
        #model.compile(optimizer="adam", loss="binary_crossentropy")
        model.compile(optimizer="adam", loss="MeanSquaredLogarithmicError")

    return model


# 예전 v2 loss 0.06
# 최종 v2 loss 0.041
# 경량인데 더 좋음 ㄷㄷ
def load_v2(path=SAVED_MODEL_PATH):
    model: tf.keras.Model
    try:
        model = tf.keras.models.load_model(path)
        print("load")
    except Exception as _:
        input_layer = tf.keras.layers.Input((125, 125, 3))
        group1_conv1_layer = tf.keras.layers.Conv2D(filters=4, kernel_size=(
            3, 3), padding="same", strides=1, activation="relu")(input_layer)
        group1_conv2_layer = tf.keras.layers.Conv2D(filters=4, kernel_size=(
            3, 3), strides=2, activation="relu")(group1_conv1_layer)
        group1_conv3_layer = tf.keras.layers.Conv2D(filters=4, kernel_size=(
            5, 5), padding="same", strides=1, activation="relu")(group1_conv2_layer)
        group1_output_layer = tf.keras.layers.BatchNormalization()(group1_conv3_layer)

        group2_conv1_layer = tf.keras.layers.Conv2D(filters=4, kernel_size=(
            3, 3), padding="same", strides=1, activation="relu")(group1_output_layer)
        group2_conv2_layer = tf.keras.layers.Conv2D(filters=4, kernel_size=(
            3, 3), strides=2, activation="relu")(group2_conv1_layer)
        group2_conv3_layer = tf.keras.layers.Conv2D(filters=4, kernel_size=(
            5, 5), padding="same", strides=1, activation="relu")(group2_conv2_layer)
        group2_output_layer = tf.keras.layers.BatchNormalization()(group2_conv3_layer)

        group3_conv1_layer = tf.keras.layers.Conv2D(filters=4, kernel_size=(
            3, 3), padding="same", strides=1, activation="relu")(group2_output_layer)
        group3_conv2_layer = tf.keras.layers.Conv2D(filters=4, kernel_size=(
            3, 3), strides=2, activation="relu")(group3_conv1_layer)
        group3_conv3_layer = tf.keras.layers.Conv2D(filters=4, kernel_size=(
            5, 5), padding="same", strides=1, activation="relu")(group3_conv2_layer)
        group3_output_layer = tf.keras.layers.BatchNormalization()(group3_conv3_layer)

        group4_upsampling_layer = tf.keras.layers.UpSampling2D()(group3_output_layer)
        group4_output_layer = tf.keras.layers.Conv2D(filters=2, kernel_size=(
            3, 3), padding="same", strides=1, activation="relu")(group4_upsampling_layer)

        group5_output_layer = tf.keras.layers.Conv2D(filters=2, kernel_size=(
            3, 3), strides=1, activation="relu")(group2_output_layer)

        group6_concat = tf.keras.layers.concatenate(
            [group4_output_layer, group5_output_layer])
        group6_conv1_layer = tf.keras.layers.Conv2D(filters=4, kernel_size=(
            3, 3), padding="same", strides=1, activation="relu")(group6_concat)

        group6_conv2_layer = tf.keras.layers.Conv2D(filters=4, kernel_size=(
            3, 3), padding="same", strides=1, activation="relu")(group6_conv1_layer)

        group6_output_layer = tf.keras.layers.Conv2D(filters=2, kernel_size=(
            3, 3), padding="same", strides=1, activation="relu")(group6_conv2_layer)

        model = tf.keras.Model(inputs=input_layer, outputs=group6_output_layer)
        model.compile(optimizer="adam", loss="MeanSquaredLogarithmicError")

    return model

## glsl 보면 gpu 연산은 vec4 계산은 파이프라인화 되어 있어 4채널 사용하는게 성능에 좋음
## 그리고 생각보다 일반적인 전략에 비해 크게 나쁘지도 않음
def create_conv_group(input_layer: tf.keras.layers.Layer):
    conv1 = tf.keras.layers.Conv2D(4, kernel_size=(
            3, 3), padding="same", strides=1, activation="relu")(input_layer)
    conv2 = tf.keras.layers.Conv2D(4, kernel_size=(
            3, 3), strides=2, activation="relu")(conv1)
    conv3 = tf.keras.layers.Conv2D(4, kernel_size=(
            5, 5), padding="same", strides=1, activation="relu")(conv2)

    return conv3

def create_down_group(input_layer):
    conv1 = tf.keras.layers.Conv2D(4, kernel_size=(
            3, 3), strides=2, activation="relu")(input_layer)
    conv2 = tf.keras.layers.Conv2D(4, kernel_size=(
            3, 3), strides=1, activation="relu")(conv1)
    
    return conv2

def create_same_group(input_layer):
    conv1 = tf.keras.layers.Conv2D(4, kernel_size=(
            3, 3), strides=1, activation="relu")(input_layer)
    return conv1

def create_up_group(input_layer):
    conv1 = tf.keras.layers.UpSampling2D()(input_layer)
    conv2 = tf.keras.layers.Conv2D(4, kernel_size=(
            3, 3), padding="same", strides=1, activation="relu")(conv1)
    
    return conv2
    

    


def load_v3(path=SAVED_MODEL_PATH):
    model: tf.keras.Model
    try:
        model = tf.keras.models.load_model(path)
        print("load")
    except Exception as _:
        input_layer = tf.keras.layers.Input((125, 125, 3))

        group1 = create_conv_group(input_layer)
        group1_output = tf.keras.layers.BatchNormalization()(group1)

        group2 = create_conv_group(group1_output)
        group2_output = tf.keras.layers.BatchNormalization()(group2)

        group3 = create_conv_group(group2_output)
        group3_output = tf.keras.layers.BatchNormalization()(group3)

        group4 = create_down_group(group1_output)
        group4_output = group4

        group5 = create_same_group(group2_output)
        group5_output = group5

        group6 = create_up_group(group3_output)
        group6_output = group6

        concat = tf.keras.layers.concatenate(
            [group4_output, group5_output, group6_output])

        conv1 = tf.keras.layers.Conv2D(4, kernel_size=(
            5, 5), padding="same", strides=1, activation="relu")(concat)

        conv2 = tf.keras.layers.Conv2D(2, kernel_size=(
            3, 3), padding="same", strides=1, activation="relu")(conv1)

        model = tf.keras.Model(inputs=input_layer, outputs=conv2)
        model.compile(optimizer="adam", loss="MeanSquaredLogarithmicError")

        return model

def apply_shape_v2(data_model: DataModel):
    data_model.input_shape = (125, 125, 3)
    data_model.output_shape = (28, 28, 2)
    data_model.pre_shape = (125, 125, 3)

def __convert_lite():
    converter = tf.lite.TFLiteConverter.from_saved_model("my-net-v2")
    tflite_model = converter.convert()

    with open("my-net-v2.tflite", "wb") as file:
        file.write(tflite_model)

def convert_lite(input_file: str, output_file: str):
    converter = tf.lite.TFLiteConverter.from_saved_model(input_file)
    tflite_model = converter.convert()

    with open(output_file, "wb") as file:
        file.write(tflite_model)


if __name__ == "__main__":
    #model = load()
    #model = load_v2("my-net-v2")
    model = load_v3()
    

    print(model.summary())
    print(model.layers[-1].dtype)
