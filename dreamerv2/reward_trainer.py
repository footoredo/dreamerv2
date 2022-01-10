import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow.keras import mixed_precision as prec

import common


class TransformerRewardPredictor(common.Module):

    def __init__(self, transformer):
        self._transformer = common.TransformerNew(output_dim=1, **transformer)

    @tf.function()
    def __call__(self, embed, action, is_first, training):  # [batch, length, *]
        x = tf.concat([embed, action], -1)
        out, weight = self._transformer(x, is_first, training)

        return tf.squeeze(out, -1), weight


class RewardTrainer:

    def __init__(self, config):
        self.config = config
        self._predictor = TransformerRewardPredictor(**config.reward_pred)
        self._encoder = common.Encoder(**config.encoder)
        self._opt = common.Optimizer('reward_predictor', **config.reward_pred_opt)

    def test(self, data):
        data = self.preprocess(data)
        embed = self._encoder(data)

        pred, weight = self._predictor(embed, data['action'], data['is_first'], training=False)

        return data, pred, weight

    @tf.function()
    def train(self, data):
        with tf.GradientTape() as tape:
            loss = self.loss(data)

        modules = [self._encoder, self._predictor]
        self._opt(tape, loss, modules)

        return loss

    @tf.function()
    def loss(self, data):
        data = self.preprocess(data)
        embed = self._encoder(data)

        pred, _ = self._predictor(embed, data['action'], data['is_first'], training=True)
        loss = ((pred - data['reward']) ** 2).mean()

        return loss


    @tf.function()
    def preprocess(self, obs):
        dtype = prec.global_policy().compute_dtype
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith('log_'):
                continue
            if value.dtype == tf.int32:
                value = value.astype(dtype)
            if value.dtype == tf.uint8:
                value = value.astype(dtype) / 255.0 - 0.5
            obs[key] = value
        obs['reward'] = {
            'identity': tf.identity,
            'sign': tf.sign,
            'tanh': tf.tanh,
        }[self.config.clip_rewards](obs['reward'])
        obs['discount'] = 1.0 - obs['is_terminal'].astype(dtype)
        obs['discount'] *= self.config.discount
        return obs
        