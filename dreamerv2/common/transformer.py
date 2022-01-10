import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow.keras.mixed_precision import experimental as prec

import common


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding

def create_look_ahead_mask(size, is_first, dtype):
    if is_first is not None:
        # masks = []
        # for i in range(size):
        #     mask0 = tf.zeros((i, i), dtype=dtype)
        #     mask1 = tf.ones((size - i, i), dtype=dtype)
        #     mask2 = tf.zeros((size, size - i), dtype=dtype)
        #     # padding = tf.tensor([[i, 0], [0, size - i]])
        #     # masks.append(tf.pad(mask, padding, "CONSTANT"))
        #     mask = tf.concat([tf.concat([mask0, mask1], 0), mask2], 1)
        #     masks.append(mask)
        ones = tf.ones((size, size), dtype=dtype)
        a = tf.linalg.band_part(ones, 0, -1)
        b = tf.linalg.band_part(ones, -1, 0) - tf.linalg.band_part(ones, 0, 0)
        mask = a[:, :, tf.newaxis] * b[:, tf.newaxis, :]
        # mask[i] is zeros where [i:, :i] is one
        # mask = 1 - tf.reduce_max(mask * tf.cast(is_first[:, tf.newaxis, tf.newaxis], dtype), 0)
        is_first = tf.linalg.diag(tf.cast(is_first, dtype))
        mask = 1 - tf.reduce_max(tf.tensordot(is_first, mask, 1), -3)
        mask = tf.expand_dims(mask, -3)  # for heads
    else:
        mask = tf.ones((size, size), dtype=dtype)
    mask = 1 - tf.linalg.band_part(mask, -1, 0)
    return tf.stop_gradient(mask)  # (seq_len, seq_len)


def create_padding_mask(seq, dtype):
    seq = tf.cast(tf.math.equal((seq ** 2).sum(-1), 0), dtype)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


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
    print("matmul_qk", matmul_qk.shape, "(..., seq_len_q, seq_len_k)")

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], dtype)
    # scaled_attention_logits = tf.tanh(matmul_qk / tf.math.sqrt(dk)) * 2.
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        # scaled_attention_logits += (mask * -1e4)
        scaled_attention_logits = mask * (-1e9) + (1. - mask) * scaled_attention_logits
    scaled_attention_logits_copy = tf.stop_gradient(tf.identity(scaled_attention_logits))
    # mask: 16 x 1 x 1 x 64
    scaled_attention_logits_norm = ((scaled_attention_logits * (1. - mask)) ** 2).sum(-1) / (1e-6 + (1. - mask).sum(-1))

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    print("attention_weights", attention_weights.shape, "(..., seq_len_q, seq_len_k)")

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    print("output", output.shape, "(..., seq_len_q, depth_v)")

    return output, scaled_attention_logits_copy, scaled_attention_logits_norm


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
        print("q", q.shape, "(batch_size, seq_len, d_model)")
        print("k", k.shape, "(batch_size, seq_len, d_model)")
        print("v", v.shape, "(batch_size, seq_len, d_model)")

        past_info = tf.concat((k, v), axis=-1)

        q = self._split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self._split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self._split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        print("q", q.shape, "(batch_size, num_heads, seq_len_q, depth)")
        print("k", k.shape, "(batch_size, num_heads, seq_len_k, depth)")
        print("v", v.shape, "(batch_size, num_heads, seq_len_v, depth)")

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights, weights_norm = scaled_dot_product_attention(q, k, v, mask, self._dtype)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        print("scaled_attention", scaled_attention.shape, "(batch_size, seq_len_q, num_heads, depth)")

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self._d_model))  # (batch_size, seq_len_q, d_model)
        print("concat_attention", concat_attention.shape, "(batch_size, seq_len_q, d_model)")

        output = self.get("dense", tfkl.Dense, self._d_model)(concat_attention)  # (batch_size, seq_len_q, d_model)
        print("output", output.shape, "(batch_size, seq_len_q, d_model)")

        print("attention_weights", attention_weights.shape)

        return output, attention_weights, weights_norm, past_info


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

        attn_output, attention_weights, weights_norm = self.get("mha", self._mha_fn)(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        # attn_output = self.get("dropout1", self._drop_fn)(attn_output, training=training)
        out1 = self.get("layernorm1", self._ln_fn)(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.get("ffn", self._ffn_fn)(out1)  # (batch_size, input_seq_len, d_model)
        # ffn_output = self.get("dropout2", self._drop_fn)(ffn_output, training=training)
        out2 = self.get("layernorm2", self._ln_fn)(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attention_weights, weights_norm


class DecoderLayer(common.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self._mha_fn = lambda: MultiHeadAttention(d_model, num_heads)
        self._ffn_fn = lambda: PointWiseFeedForwardNetwork(d_model, dff)
        self._drop_fn = lambda: tfkl.Dropout(rate)
        self._ln_fn = lambda: tfkl.LayerNormalization(epsilon=1e-6)

    @tf.function
    def __call__(self, x, enc_output, training, padding_mask):
        # x.shape == (batch_size, d_model)
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # attn1, attn_weights_block1 = self.get("mha1", self._mha_fn)(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        # attn1 = self.get("dropout1", self._drop_fn)(attn1, training=training)
        # out1 = self.get("layernorm1", self._ln_fn)(attn1 + x)
        attn_weights_block1 = None
        out1 = x[:, tf.newaxis, :]

        attn2, attn_weights_block2, weights_norm2 = self.get("mha2", self._mha_fn)(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.get("dropout2", self._drop_fn)(attn2, training=training)
        out2 = self.get("layernorm2", self._ln_fn)(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.get("ffn", self._ffn_fn)(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.get("dropout3", self._drop_fn)(ffn_output, training=training)
        out3 = self.get("layernorm3", self._ln_fn)(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3[:, 0, :], attn_weights_block1, attn_weights_block2[:, :, 0, :], weights_norm2[:, :, 0]


class Encoder(common.Module):
    def __init__(self, num_layers, d_model, num_heads, dff,
                maximum_position_encoding, rate=0.1, dtype=None):
        super().__init__()
        self._d_model = d_model
        self._num_layers = num_layers
        self._dtype = dtype or prec.global_policy().compute_dtype

        self._pe = positional_encoding(maximum_position_encoding, d_model)
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
        # pos_encoding = self.get('pos_encoding', self._pe_fn)
        pe = tf.constant(self._pe, self._dtype)
        x += pe[:, :seq_len, :]

        # x = self.get("dropout", self._drop_fn)(x, training=training)

        attention_weights = dict()
        weights_norm = dict()

        for i in range(self._num_layers):
            x, w, n = self.get(f"enc{i}", self._enc_fn)(x, training, mask)
            attention_weights[f"enc{i}"] = tf.stop_gradient(w)
            weights_norm[f"enc{i}"] = tf.stop_gradient(n)

        return x, attention_weights, weights_norm  # (batch_size, input_seq_len, d_model)


class Decoder(common.Module):
    def __init__(self, num_layers, d_model, num_heads, dff,
                maximum_position_encoding, no_pe=False, reverse_pe=False, rate=0.1, dtype=None):
        super().__init__()
        self._d_model = d_model
        self._num_layers = num_layers
        self._dtype = dtype or prec.global_policy().compute_dtype

        self._pe = positional_encoding(maximum_position_encoding, d_model)
        self._no_pe = no_pe
        self._reverse_pe = reverse_pe

        self._dec_fn = lambda: DecoderLayer(d_model, num_heads, dff, rate)
        self._drop_fn = lambda: tfkl.Dropout(rate)

    @tf.function
    def __call__(self, x, enc_output, training, padding_mask):
        # (batch_size, target_seq_len, d_model)

        seq_len = tf.shape(enc_output)[1]
        x *= tf.math.sqrt(tf.cast(self._d_model, self._dtype))
        enc_output *= tf.math.sqrt(tf.cast(self._d_model, self._dtype))
        pe = tf.constant(self._pe, self._dtype)
        if not self._no_pe:
            if self._reverse_pe:
                enc_output += tf.reverse(pe[:, :seq_len, :], [1])
            else:
                enc_output += pe[:, :seq_len, :]

        attention_weights = dict()
        weights_norm = dict()

        for i in range(self._num_layers):
            x, w1, w2, n2 = self.get(f"dec{i}", self._dec_fn)(x, enc_output, training, padding_mask)
            print(i, "w2", w2.shape)
            attention_weights[f"dec{i}"] = tf.stop_gradient(w2)
            weights_norm[f"dec{i}"] = tf.stop_gradient(n2)

        return x, attention_weights, weights_norm


class TransformerOld(common.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, rate=0.1, dtype=None):
        super().__init__()
        self._dtype = dtype or prec.global_policy().compute_dtype
        self._d_model = d_model

        self._encoder_fn = lambda: Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)

    @tf.function
    def __call__(self, inp, ctx, training):

        # look_ahead_mask = create_look_ahead_mask(tf.shape(inp)[1], self._dtype)
        padding_mask = tf.stop_gradient(create_padding_mask(inp, self._dtype))
        x = self.get("dense", tfkl.Dense, self._d_model)(inp)
        x = self.get("norm", tfkl.LayerNormalization, epsilon=1e-6)(x)
        # padding_mask = None
        output, attention_weights = self.get("encoder", self._encoder_fn)(x, training, padding_mask)  # (batch_size, inp_seq_len, d_model)

        return output, attention_weights, padding_mask


class Transformer(common.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, output_dim, no_pe=False, reverse_pe=False, rate=0.1, dtype=None):
        super().__init__()
        self._dtype = dtype or prec.global_policy().compute_dtype
        self._d_model = d_model
        self._output_dim = output_dim

        self._dec_fn = lambda: Decoder(num_layers, d_model, num_heads, dff, pe_input, no_pe, reverse_pe, rate)

    @tf.function
    def __call__(self, inp, ctx, training):
        padding_mask = tf.stop_gradient(create_padding_mask(inp, self._dtype))
        x = self.get("dense1", tfkl.Dense, self._d_model, activation='relu')(inp)
        # x = self.get("norm1", tfkl.LayerNormalization, epsilon=1e-6)(x)

        y = self.get("dense2", tfkl.Dense, self._d_model, activation='relu')(ctx)
        # y = self.get("norm2", tfkl.LayerNormalization, epsilon=1e-6)(y)

        output, attention_weights, weights_norm = self.get("decoder", self._dec_fn)(y, x, training, padding_mask)  # (batch_size, inp_seq_len, d_model)

        output = self.get("dense3", tfkl.Dense, self._output_dim)(output)
        output = self.get("norm3", tfkl.LayerNormalization, epsilon=1e-6)(output)

        return output, attention_weights, weights_norm, padding_mask


class DecoderLayerNew(common.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self._mha_fn = lambda: MultiHeadAttention(d_model, num_heads)
        self._ffn_fn = lambda: PointWiseFeedForwardNetwork(d_model, dff)
        self._drop_fn = lambda: tfkl.Dropout(rate)
        self._ln_fn = lambda: tfkl.LayerNormalization(epsilon=1e-6)

    @tf.function
    def __call__(self, x, training, look_ahead_mask):
        # x.shape == (batch_size, target_seq_len, d_model)
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1, _, past_info = self.get("mha1", self._mha_fn)(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.get("dropout1", self._drop_fn)(attn1, training=training)
        out1 = self.get("layernorm1", self._ln_fn)(attn1 + x)  # (batch_size, target_seq_len, d_model)

        # attn2, attn_weights_block2, weights_norm2 = self.get("mha2", self._mha_fn)(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        # attn2 = self.get("dropout2", self._drop_fn)(attn2, training=training)
        # out2 = self.get("layernorm2", self._ln_fn)(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.get("ffn", self._ffn_fn)(out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.get("dropout3", self._drop_fn)(ffn_output, training=training)
        out3 = self.get("layernorm3", self._ln_fn)(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, past_info


class DecoderNew(common.Module):
    def __init__(self, num_layers, d_model, num_heads, dff,
                maximum_position_encoding, no_pe=False, reverse_pe=False, rate=0.1, dtype=None):
        super().__init__()
        self._d_model = d_model
        self._num_layers = num_layers
        self._dtype = dtype or prec.global_policy().compute_dtype

        self._pe = positional_encoding(maximum_position_encoding, d_model)
        self._no_pe = no_pe
        self._reverse_pe = reverse_pe

        self._dec_fn = lambda: DecoderLayerNew(d_model, num_heads, dff, rate)
        self._drop_fn = lambda: tfkl.Dropout(rate)

    @tf.function
    def __call__(self, x, training, look_ahead_mask):
        # x: (batch_size, target_seq_len, ?)

        x = self.get('input_dense', tfkl.Dense, self._d_model)(x)
        x = self.get('input_norm', common.NormLayer, 'layer')(x)

        seq_len = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self._d_model, self._dtype))
        pe = tf.constant(self._pe, self._dtype)
        if not self._no_pe:
            if self._reverse_pe:
                x += tf.reverse(pe[:, :seq_len, :], [1])
            else:
                x += pe[:, :seq_len, :]

        attention_weights = dict()

        for i in range(self._num_layers):
            x, w, _ = self.get(f"dec{i}", self._dec_fn)(x, training, look_ahead_mask)
            print(i, "w", w.shape)
            attention_weights[f"dec{i}"] = tf.stop_gradient(w)
            # weights_norm[f"dec{i}"] = tf.stop_gradient(n2)

        return x, attention_weights


class TransformerNew(common.Module):

    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, output_dim, rate=0.1, dtype=None):
        super().__init__()

        self._dtype = dtype or prec.global_policy().compute_dtype
        self._d_model = d_model
        self._output_dim = output_dim

        self._dec_fn = lambda: DecoderNew(num_layers, d_model, num_heads, dff, pe_input, False, False, rate)

    @tf.function
    def __call__(self, x, is_first, training):
        # x: (batch_size, tar_seq_len, ?)
        # is_first: (batch_size, tar_seq_len)

        print("x.shape", x.shape, flush=True)
        print("is_first.shape", is_first.shape, flush=True)

        look_ahead_mask = create_look_ahead_mask(tf.shape(x)[1], is_first, self._dtype)

        dec_output, attention_weights = self.get('decoder', self._dec_fn)(x, training, look_ahead_mask)
        print('dec_output', dec_output.shape, flush=True)

        output = self.get("dense", tfkl.Dense, self._output_dim)(dec_output)
        print('output', output.shape, flush=True)

        return output, attention_weights


def transformer_test():
    z = tf.constant([1, 0, 1, 0, 0, 1], dtype=tf.float32)
    mask = create_look_ahead_mask(tf.shape(z)[-1], z, tf.float32)
    print(mask)

    # tf.Tensor(
    # [[0. 1. 1. 1. 1. 1.]
    # [0. 0. 1. 1. 1. 1.]
    # [1. 1. 0. 1. 1. 1.]
    # [1. 1. 0. 0. 1. 1.]
    # [1. 1. 0. 0. 0. 1.]
    # [1. 1. 1. 1. 1. 0.]], shape=(6, 6), dtype=float32)


    z = tf.constant([[1, 0, 1, 0, 0, 1], [1, 1, 0, 0, 1, 0]], dtype=tf.float32)
    mask = create_look_ahead_mask(tf.shape(z)[-1], z, tf.float32)
    print(mask.shape)
    print(mask[0])
    print(mask[1])

    # (2, 6, 6)
    # tf.Tensor(
    # [[0. 1. 1. 1. 1. 1.]
    # [0. 0. 1. 1. 1. 1.]
    # [1. 1. 0. 1. 1. 1.]
    # [1. 1. 0. 0. 1. 1.]
    # [1. 1. 0. 0. 0. 1.]
    # [1. 1. 1. 1. 1. 0.]], shape=(6, 6), dtype=float32)
    # tf.Tensor(
    # [[0. 1. 1. 1. 1. 1.]
    # [1. 0. 1. 1. 1. 1.]
    # [1. 0. 0. 1. 1. 1.]
    # [1. 0. 0. 0. 1. 1.]
    # [1. 1. 1. 1. 0. 1.]
    # [1. 1. 1. 1. 0. 0.]], shape=(6, 6), dtype=float32)

    z = tf.constant([[[1, 0, 1, 0, 0, 1], [1, 1, 0, 0, 1, 0]]], dtype=tf.float32)
    mask = create_look_ahead_mask(tf.shape(z)[-1], z, tf.float32)
    print(mask.shape)
    print(mask[0, 0])
    print(mask[0, 1])

    # (1, 2, 6, 6)
    # tf.Tensor(
    # [[0. 1. 1. 1. 1. 1.]
    # [0. 0. 1. 1. 1. 1.]
    # [1. 1. 0. 1. 1. 1.]
    # [1. 1. 0. 0. 1. 1.]
    # [1. 1. 0. 0. 0. 1.]
    # [1. 1. 1. 1. 1. 0.]], shape=(6, 6), dtype=float32)
    # tf.Tensor(
    # [[0. 1. 1. 1. 1. 1.]
    # [1. 0. 1. 1. 1. 1.]
    # [1. 0. 0. 1. 1. 1.]
    # [1. 0. 0. 0. 1. 1.]
    # [1. 1. 1. 1. 0. 1.]
    # [1. 1. 1. 1. 0. 0.]], shape=(6, 6), dtype=float32)