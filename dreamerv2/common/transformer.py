import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow.keras.mixed_precision import experimental as prec

import common


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model, dtype):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=dtype)


def create_look_ahead_mask(size, dtype):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size), dtype=dtype), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask, dtype):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], dtype)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(common.Module):
    def __init__(self, d_model, num_heads, dtype=None):
        super().__init__()
        self._num_heads = num_heads
        self._d_model = d_model
        assert d_model % num_heads == 0
        self._depth = d_model // num_heads
        self._dtype = dtype or prec.global_policy().compute_dtype

    def _split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self._num_heads, self._depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    @tf.function
    def __call__(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.get("wq", tfkl.Dense, self._d_model)(q)   # (batch_size, seq_len, d_model)
        k = self.get("wk", tfkl.Dense, self._d_model)(k)   # (batch_size, seq_len, d_model)
        v = self.get("wv", tfkl.Dense, self._d_model)(v)   # (batch_size, seq_len, d_model)

        q = self._split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self._split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self._split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask, self._dtype)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self._d_model))  # (batch_size, seq_len_q, d_model)

        output = self.get("dense", tfkl.Dense, self._d_model)(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


# def point_wise_feed_forward_network(d_model, dff):
#   return tf.keras.Sequential([
#       tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
#       tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
#   ])

class PointWiseFeedForwardNetwork(common.Module):
    def __init__(self, d_model, dff):
        self._d_model = d_model
        self._dff = dff

    @tf.function
    def __call__(self, x):
        out1 = self.get("dense1", tfkl.Dense, self._dff, activation='relu')(x)
        out2 = self.get("dense2", tfkl.Dense, self._d_model)(out1)
        return out2


class EncoderLayer(common.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self._mha_fn = lambda: MultiHeadAttention(d_model, num_heads)
        self._ffn_fn = lambda: PointWiseFeedForwardNetwork(d_model, dff)
        self._drop_fn = lambda: tfkl.Dropout(rate)
        self._ln_fn = lambda: tfkl.LayerNormalization(epsilon=1e-6)

    @tf.function
    def __call__(self, x, training, mask):

        attn_output, attention_weights = self.get("mha", self._mha_fn)(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.get("dropout1", self._drop_fn)(attn_output, training=training)
        out1 = self.get("layernorm1", self._ln_fn)(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.get("ffn", self._ffn_fn)(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.get("dropout2", self._drop_fn)(ffn_output, training=training)
        out2 = self.get("layernorm2", self._ln_fn)(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attention_weights


class DecoderLayer(common.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self._mha_fn = lambda: MultiHeadAttention(d_model, num_heads)
        self._ffn_fn = lambda: PointWiseFeedForwardNetwork(d_model, dff)
        self._drop_fn = lambda: tfkl.Dropout(rate)
        self._ln_fn = lambda: tfkl.LayerNormalization(epsilon=1e-6)

    @tf.function
    def __call__(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.get("mha1", self._mha_fn)(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.get("dropout1", self._drop_fn)(attn1, training=training)
        out1 = self.get("layernorm1", self._ln_fn)(attn1 + x)

        attn2, attn_weights_block2 = self.get("mha2", self._mha_fn)(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.get("dropout2", self._drop_fn)(attn2, training=training)
        out2 = self.get("layernorm2", self._ln_fn)(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.get("ffn", self._ffn_fn)(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.get("dropout3", self._drop_fn)(ffn_output, training=training)
        out3 = self.get("layernorm3", self._ln_fn)(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(common.Module):
    def __init__(self, num_layers, d_model, num_heads, dff,
                maximum_position_encoding, rate=0.1, dtype=None):
        super().__init__()
        self._d_model = d_model
        self._num_layers = num_layers
        self._dtype = dtype or prec.global_policy().compute_dtype

        self._pos_encoding = positional_encoding(maximum_position_encoding, d_model, self._dtype)
        self._enc_fn = lambda: EncoderLayer(d_model, num_heads, dff, rate)
        self._drop_fn = lambda: tfkl.Dropout(rate)

        # self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
        #                 for _ in range(num_layers)]

        # self.dropout = tf.keras.layers.Dropout(rate)

    @tf.function
    def __call__(self, x, training, mask):
        # x: (batch_size, input_seq_len, d_model)

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x *= tf.math.sqrt(tf.cast(self._d_model, self._dtype))
        x += self._pos_encoding[:, :seq_len, :]

        x = self.get("dropout", self._drop_fn)(x, training=training)
        # x = self.dropout(x, training=training)

        attention_weights = dict()

        for i in range(self._num_layers):
            x, w = self.get(f"enc{i}", self._enc_fn)(x, training, mask)
            attention_weights[f"enc{i}"] = w
            # x = self.enc_layers[i](x, training, mask)

        return x, attention_weights  # (batch_size, input_seq_len, d_model)


# class Decoder(common.Module):
#     def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
#                 maximum_position_encoding, rate=0.1, dtype=None):
#         super().__init__()
#         self._d_model = d_model
#         self._num_layers = num_layers
#         self._dtype = dtype or prec.global_policy().compute_dtype

#         self._pos_encoding = positional_encoding(maximum_position_encoding, d_model, dtype)

#         self._enc_fn = lambda: DecoderLayer(d_model, num_heads, dff, rate)
#         self._drop_fn = lambda: tfkl.Dropout(rate)

#     @tf.function
#     def __call__(self, x, enc_output, training, look_ahead_mask, padding_mask):
#         # (batch_size, target_seq_len, d_model)

#         seq_len = tf.shape(x)[1]
#         attention_weights = {}

#         x = self.embedding(x)  
#         x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#         x += self.pos_encoding[:, :seq_len, :]

#         x = self.dropout(x, training=training)

#         for i in range(self.num_layers):
#             x, block1, block2 = self.dec_layers[i](x, enc_output, training,
#                                                 look_ahead_mask, padding_mask)

#         attention_weights[f'decoder_layer{i+1}_block1'] = block1
#         attention_weights[f'decoder_layer{i+1}_block2'] = block2

#         # x.shape == (batch_size, target_seq_len, d_model)
#         return x, attention_weights


class Transformer(common.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, rate=0.1, dtype=None):
        super().__init__()
        self._dtype = dtype or prec.global_policy().compute_dtype

        self._encoder_fn = lambda: Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)

    @tf.function
    def __call__(self, inp, training):

        # look_ahead_mask = create_look_ahead_mask(tf.shape(inp)[1], self._dtype)
        output, attention_weights = self.get("encoder", self._encoder_fn)(inp, training, None)  # (batch_size, inp_seq_len, d_model)

        return output, attention_weights
