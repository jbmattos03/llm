import tensorflow as tf
from tensorflow.keras.layers import Dense

class PositionFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(PositionFeedForward, self).__init__()
        self.d_model = d_model
        self.dff = dff
        self.dense1 = Dense(dff, activation="relu")
        self.dense2 = Dense(d_model)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)

        return x