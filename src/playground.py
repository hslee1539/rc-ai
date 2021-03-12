import tensorflow as tf
import os

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=5, kernel_size=(15, 15), activation="relu"))

model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(15, 15), activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(15, 15), activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=(15, 15), activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
model.build((1,256,256,3))

print(model.summary())