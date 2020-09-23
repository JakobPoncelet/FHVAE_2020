import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from .transf_layers import *
from .transf_utils import *

class TransformerEncoder(tf.keras.Model):  # instead of layers.Layer
    def __init__(self, num_layers, d_model, num_heads, dff, pe_max_len, name, rate=0.1, bs=256):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.rate = rate
        self.bs = bs

        self.pos_encoding = positional_encoding(pe_max_len, self.d_model)

        self.input_proj = tf.keras.models.Sequential(name='en_proj')
        self.input_proj.add(layers.Dense(units=self.d_model, kernel_initializer='glorot_normal'))
        self.input_proj.add(layers.LayerNormalization(epsilon=1e-6))

        self.dropout = layers.Dropout(rate, name='en_proj_dp')

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, 'EN'+str(_), rate)
                           for _ in range(num_layers)]
        self.fixed_attn = FixedAttention(d_model, num_heads, 'FA', rate, bs)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # doing projection (instead of embedding) and adding position encoding.
        x = self.input_proj(x)  # (batch_size, input_seq_len, d_model)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        pos_enc = self.pos_encoding[:, :seq_len, :]

        x += tf.cast(pos_enc, x.dtype)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)  # (batch_size, input_seq_len, d_model)

        # one last attention layer with query of fixed size to remove the time dimension
        out = self.fixed_attn(x, training, mask)  # (batch_size, d_model)

        return out
