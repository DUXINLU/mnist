import tensorflow as tf


class MNISTMLPModel(tf.keras.Model):
    def __init__(self):
        super(MNISTMLPModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.output_layer = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        return self.output_layer(x)
