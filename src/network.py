import tensorflow as tf
import os

"""
1. 모델 생성(model.h5 가 없는 경우)
2. 모델 로딩(model.h5 가 있는 경우)
3. loadModel() : tensorflow.keras.model.Model 제공
4. saveModel(model : tensorflow.keras.model.Model) 제공
"""

SAVED_MODEL_PATH = os.path.dirname(
    __file__) + "/model"  # 경로 중 디렉토리명만 얻고 model 폴더 설정


def load():
    model: tf.keras.Model

    try:
        model = tf.keras.models.load_model(SAVED_MODEL_PATH)
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
            filters=17, kernel_size=(13, 13), strides=2, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Conv2D(
            filters=2, kernel_size=(5,5), strides=1,activation="relu"))
        model.build((1, 256, 256, 3))
        #model.compile(optimizer="adam", loss="binary_crossentropy")
        model.compile(optimizer="adam", loss="MeanSquaredLogarithmicError")

    return model


if __name__ == "__main__":
    model = load()
    print(model.summary())
