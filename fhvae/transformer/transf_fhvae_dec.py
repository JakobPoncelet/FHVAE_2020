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
        # self.input_proj.add(layers.Dropout(rate=dp))
        self.input_proj.add(layers.LayerNormalization(epsilon=1e-6))

        self.dropout = layers.Dropout(rate, name='en_proj_dp')

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, 'EN'+str(_), rate)
                           for _ in range(num_layers)]
        self.fixed_attn = FixedAttention(d_model, num_heads, 'FA', rate, bs)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        #x = tf.reshape(x,[x.shape[0],x.shape[1],-1]) # B*T*(D*n)  flatten on channels

        # doing projection (instead of embedding) and adding position encoding.
        x = self.input_proj(x)  # (batch_size, input_seq_len, d_model)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        pos_enc = self.pos_encoding[:, :seq_len, :]

        x += tf.cast(pos_enc, x.dtype)

        # print('dropout.rate: ',str(self.dropout.rate))
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)  # (batch_size, input_seq_len, d_model)

        # one last attention layer with query of fixed size to remove the time dimension
        out = self.fixed_attn(x, training, mask)  # (batch_size, d_model)

        return out

class TransformerDecoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, pe_max_len, name, rate=0.1, bs=256):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.rate = rate
        self.bs = bs

        self.pos_encoding = positional_encoding(pe_max_len, self.d_model)

        self.input_proj = tf.keras.models.Sequential(name='dec_proj')
        self.input_proj.add(layers.Dense(units=self.d_model, kernel_initializer='glorot_normal'))
        self.input_proj.add(layers.LayerNormalization(epsilon=1e-6))

        self.dropout = layers.Dropout(rate, name='dec_proj_dp')

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, 'DEC' + str(_), rate)
                           for _ in range(num_layers)]


    def call(self, x, z1, z2, look_ahead_mask, padding_mask, training):

        bs, T = tf.shape(input=x)[0], tf.shape(input=x)[1]

        z1 = tf.tile(tf.expand_dims(z1, 1), (1, T, 1))
        z2 = tf.tile(tf.expand_dims(z2, 1), (1, T, 1))
        z1_z2 = tf.concat([z1, z2], axis=-1)

        z1_z2 = self.input_proj()

        #x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        #x *= tf.math.rsqrt(tf.cast(self.d_model, tf.float32))
        #x += tf.cast(self.pos_encoding[:, :seq_len, :],x.dtype)



        # x = self.dropout(x, training=training)

        attention_weights = {}
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, z1_z2, training,
                                                   look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # before softmax
        x = self.final_layer(x)

        # x.shape == (batch_size, target_seq_len, target_vocab_size if proj else d_model)
        return x, attention_weights