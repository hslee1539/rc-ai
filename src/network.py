import tensorflow as tf
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


def load(path = SAVED_MODEL_PATH):
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
            filters=2, kernel_size=(5,5), strides=1,activation="relu"))
        model.build((1, 100, 100, 3))
        #model.compile(optimizer="adam", loss="binary_crossentropy")
        model.compile(optimizer="adam", loss="MeanSquaredLogarithmicError")

    return model

def load_v2(path = SAVED_MODEL_PATH):
    model: tf.keras.Model
    try:
        model = tf.keras.models.load_model(path)
        print("load")
    except Exception as e:
        input_layer = tf.keras.layers.Input((125, 125, 3))
        group1_conv1_layer = tf.keras.layers.Conv2D(filters=5, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(input_layer)
        group1_conv2_layer = tf.keras.layers.Conv2D(filters=5, kernel_size=(3, 3), strides=2, activation="relu")(group1_conv1_layer)
        group1_output_layer = tf.keras.layers.BatchNormalization()(group1_conv2_layer)

        group2_conv1_layer = tf.keras.layers.Conv2D(filters=13, kernel_size=(3, 3), padding="same",strides=1, activation="relu")(group1_output_layer)
        group2_conv2_layer = tf.keras.layers.Conv2D(filters=13, kernel_size=(3, 3), strides=2, activation="relu")(group2_conv1_layer)
        group2_output_layer = tf.keras.layers.BatchNormalization()(group2_conv2_layer)

        group3_conv1_layer = tf.keras.layers.Conv2D(filters=23, kernel_size=(3, 3), padding="same",strides=1, activation="relu")(group2_output_layer)
        group3_conv2_layer = tf.keras.layers.Conv2D(filters=23, kernel_size=(3, 3), strides=2, activation="relu")(group3_conv1_layer)
        group3_output_layer = tf.keras.layers.BatchNormalization()(group3_conv2_layer)
        
        group4_upsampling_layer = tf.keras.layers.UpSampling2D()(group3_output_layer)
        group4_output_layer = tf.keras.layers.Conv2D(filters=23, kernel_size=(3, 3), padding="same",strides=1, activation="relu")(group4_upsampling_layer)

        group5_output_layer = tf.keras.layers.Conv2D(filters=13, kernel_size=(3, 3) ,strides=1, activation="relu")(group2_output_layer)

        group6_concat = tf.keras.layers.concatenate([group4_output_layer, group5_output_layer])
        group6_output_layer = tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 3),strides=1, activation="relu")(group6_concat)

        model = tf.keras.Model(inputs=input_layer, outputs=group6_output_layer)
        model.compile(optimizer="adam", loss="MeanSquaredLogarithmicError")
        
    return model

def apply_shape_v2(data_model: DataModel):
    data_model.input_shape = (125, 125, 3)
    data_model.output_shape = (28, 28, 2)
    data_model.pre_shape = (125, 125, 3)





if __name__ == "__main__":
    #model = load()
    model = load_v2()
    
    print(model.summary())
