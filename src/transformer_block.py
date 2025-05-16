from operator import le
from multihead_attention import MultiHeadAttention
from positionwise_feed_forward import PositionFeedForward

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, LayerNormalization
from tensorflow.keras.models import Model
import numpy as np

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionFeedForward(d_model, dff)
        self.layernorm1 =LayerNormalization(epsilon=le-6) 
        self.layernorm2 = LayerNormalization(epsilon=le-6) 
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training, mask):
        attention_output = self