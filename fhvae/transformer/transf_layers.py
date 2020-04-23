import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from .transf_utils import *


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, name, rate=0.1):
        super(EncoderLayer, self).__init__(name=name)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name=name+'_LN1')
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name=name+'_LN2')

        self.dropout1 = layers.Dropout(rate, name=name+'_dp1')
        self.dropout2 = layers.Dropout(rate, name=name+'_dp2')

    def call(self, input, training, mask):
        attn_output, slf_attn_weight = self.mha(input, input, input, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(input + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, name, rate=0.1):
        super(DecoderLayer, self).__init__(name=name)

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # POSSIBLY DELETE THE EPSILON HERE!!
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name=name+'_LN1')
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name=name+'_LN2')
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6, name=name+'_LN3')

        self.dropout1 = layers.Dropout(rate, name=name+'_dp1')
        self.dropout2 = layers.Dropout(rate, name=name+'_dp2')
        self.dropout3 = layers.Dropout(rate, name=name+'_dp3')

    def call(self, inputs, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(inputs, inputs, inputs, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + inputs)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

class FixedAttention(layers.Layer):
    def __init__(self, d_model, num_heads, name, rate=0.1, bs=256):
        super(FixedAttention, self).__init__()

        self.fixed_query = tf.Variable(tf.random.normal([bs, 1, d_model], stddev=1.0))
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout = layers.Dropout(rate, name=name+"_dp_fixed")
        self.layernorm = layers.LayerNormalization(epsilon=1e-6, name=name+'_LN_fixed')

    def call(self, input, training, mask):
        # the last batch is not complete (input.shape[0] < bs)
        # query = tf.slice(self.fixed_query, [0, 0, 0], [tf.shape(input)[0], -1, -1])
        query = self.fixed_query[:tf.shape(input)[0], :, :]
        attn, attn_weights = self.mha(input, input, query, mask)  # last batch is not complete
        attn = self.dropout(attn, training=training)
        out = self.layernorm(attn)

        out = tf.squeeze(out, axis=1)

        return out

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads  # e.g. 64

        init = tf.keras.initializers.RandomNormal(mean=0, stddev=np.sqrt(2.0 / (d_model + self.depth)))
        #init = tf.keras.initializers.glorot_normal()

        self.wq = layers.Dense(d_model, kernel_initializer=init)  # (feature_in_dim, d_model)
        self.wk = layers.Dense(d_model, kernel_initializer=init)
        self.wv = layers.Dense(d_model, kernel_initializer=init)

        self.dense = layers.Dense(d_model, kernel_initializer='glorot_normal')  # (feature_in_dim, d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  # -1 for seq_len,
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        '''
        :param v: input data , shape(batch_size, seq_len, feature_in_dim)
        :param k: input data , shape(batch_size, seq_len, feature_in_dim)
        :param q: input data , shape(batch_size, seq_len, feature_in_dim)
        :param mask: padding mask, shape(batchsize, 1, 1, seq_len) # 1 for broadcast
        :return:
        '''
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
