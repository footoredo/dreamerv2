import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import common


class EnsembleRSSM(common.Module):

    def __init__(
            self, ensemble=5, stoch=30, deter=200, hidden=200, discrete=False,
            act='elu', norm='none', std_act='softplus', min_std=0.1, seq_model='rnn',
            num_actions=None, transformer=None):
        super().__init__()
        self._ensemble = ensemble
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._act = get_act(act)
        self._norm = norm
        self._std_act = std_act
        self._min_std = min_std
        self._seq_model = seq_model
        self._num_actions = num_actions
        print("seq_model:", seq_model)

        # rnn state: {"stoch": [batch_size, stoch, discrete], "deter": [batch_size, deter]}
        # transformer state: {"stoch": [batch_size, stoch, discrete], "t_stoch": [batch_size, seq_len - 1, stoch, discrete], 
        #                     "t_action": [batch_size, seq_len - 1, n_action], "deter": [batch_size, deter]}
    
        self._cell = GRUCell(self._deter, norm=True) if seq_model == "rnn" else None
        self._memory_size = transformer.pop("memory_size")
        self._transformer = common.Transformer(d_model=hidden, **transformer) if seq_model == "transformer" else None
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        if self._discrete:
            stoch_size = [self._stoch, self._discrete]
            state = dict(
                logit=tf.zeros([batch_size] + stoch_size, dtype),
                stoch=tf.zeros([batch_size] + stoch_size, dtype))
        else:
            stoch_size = [self._stoch]
            state = dict(
                mean=tf.zeros([batch_size] + stoch_size, dtype),
                std=tf.zeros([batch_size] + stoch_size, dtype),
                stoch=tf.zeros([batch_size] + stoch_size, dtype))
        if self.use_rnn:
            state['deter'] = self._cell.get_initial_state(None, batch_size, dtype)
        elif self.use_transformer:
            state['deter'] = tf.zeros([batch_size, self._deter], dtype)
            state['t_stoch'] = tf.zeros([batch_size, self._memory_size] + stoch_size, dtype)
            state['t_action'] = tf.zeros([batch_size, self._memory_size, self._num_actions], dtype)
            # state['t_len'] = tf.zeros([batch_size, 1], dtype=tf.int32)
        return state

    @property
    def use_rnn(self):
        return self._seq_model == 'rnn'

    def use_transformer(self):
        return self._seq_model == 'transformer'

    @tf.function
    def observe(self, embed, action, is_first, training, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0])
        post, prior = common.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs, training),
            (swap(action), swap(embed), swap(is_first)), (state, state))
        # for k, v in post.items():
        #     print(k, type(v).__name__)
        def _swap(k, v):
            if k.startswith("t_"):
                return swap(v)
            else:
                return swap(v)
        post = {k: _swap(k, v) for k, v in post.items()}
        prior = {k: _swap(k, v) for k, v in prior.items()}
        return post, prior

    @tf.function
    def imagine(self, action, training, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0])
        assert isinstance(state, dict), state
        action = swap(action)
        prior = common.static_scan(lambda prev, inputs: self.img_step(prev, inputs, training), action, state)
        def _swap(k, v):
            print(k, v.shape)
            if k.startswith("t_"):
                return swap(v)
            else:
                return swap(v)
        prior = {k: _swap(k, v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = self._cast(state['stoch'])
        if self._discrete:
            shape = stoch.shape[:-2] + [self._stoch * self._discrete]
            stoch = tf.reshape(stoch, shape)
        return tf.concat([stoch, state['deter']], -1)

    def get_dist(self, state, ensemble=False):
        if ensemble:
            state = self._suff_stats_ensemble(state['deter'])
        if self._discrete:
            logit = state['logit']
            logit = tf.cast(logit, tf.float32)
            dist = tfd.Independent(common.OneHotDist(logit), 1)
        else:
            mean, std = state['mean'], state['std']
            mean = tf.cast(mean, tf.float32)
            std = tf.cast(std, tf.float32)
            dist = tfd.MultivariateNormalDiag(mean, std)
        return dist

    @tf.function
    def obs_step(self, prev_state, prev_action, embed, is_first, training, sample=True):
        # if is_first.any():
        print(list(prev_state.keys()), flush=True)
        # prev_state, prev_action = tf.nest.map_structure(
        #     lambda x: tf.einsum(
        #         'b,b...->b...', 1.0 - is_first.astype(x.dtype), x),
        #     (prev_state, prev_action))
        mask_fn = lambda x: tf.einsum('b,b...->b...', 1.0 - is_first.astype(x.dtype), x)
        prev_action = mask_fn(prev_action)
        for k in prev_state.keys():
            # if not k.startswith('t_'):
            prev_state[k] = mask_fn(prev_state[k])
        prior = self.img_step(prev_state, prev_action, training, sample)
        x = tf.concat([prior['deter'], embed], -1)
        x = self.get('obs_out', tfkl.Dense, self._hidden)(x)
        x = self.get('obs_out_norm', NormLayer, self._norm)(x)
        x = self._act(x)
        stats = self._suff_stats_layer('obs_dist', x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
        if self.use_transformer:
            post['t_stoch'] = tf.identity(prior['t_stoch'])
            post['t_action'] = tf.identity(prior['t_action'])
            # post['t_len'] = tf.identity(prior['t_len'])
        return post, prior

    @tf.function
    def img_step(self, prev_state, prev_action, training, sample=True):
        if self.use_rnn:
            prev_stoch = self._cast(prev_state['stoch'])
            prev_action = self._cast(prev_action)
        elif self.use_transformer:
            # cur_stoch = tf.expand_dims(self._cast(prev_state['stoch']), 1)
            # cur_action = tf.expand_dims(self._cast(prev_action), 1)

            # pos = prev_state['t_len'] % self._memory_size  # [batch, 1]
            # print(prev_state['t_len'], type(prev_state['t_stoch']).__name__)
            h_stoch = self._cast(tf.identity(prev_state['t_stoch']))
            h_action = self._cast(tf.identity(prev_state['t_action']))

            h_stoch = tf.roll(h_stoch, shift=-1, axis=1)
            h_action = tf.roll(h_action, shift=-1, axis=1)

            bs, sl = tf.shape(h_stoch)[:2]
            indices = tf.stack([tf.range(bs), tf.ones(bs, dtype=tf.int32) * (sl - 1)], 1)
            print(indices)
            h_stoch = tf.tensor_scatter_nd_update(h_stoch, indices, self._cast(prev_state['stoch']))
            h_action = tf.tensor_scatter_nd_update(h_action, indices, self._cast(prev_action))
            # h_stoch[:, -1].assign(self._cast(prev_state['stoch']))
            # h_action[:, -1].assign(self._cast(prev_action))

            # h_stoch[pos] = self._cast(prev_state['stoch'])
            # h_action[pos] = self._cast(prev_action)
            # pos = tf.concat([tf.expand_dims(tf.range(pos.shape[0]), -1), pos], -1)
            # h_stoch = tf.tensor_scatter_nd_update(h_stoch, pos, self._cast(prev_state['stoch']))
            # h_action = tf.tensor_scatter_nd_update(h_action, pos, self._cast(prev_action))
            # h_stoch[:, pos].assign(self._cast(prev_state['stoch']))
            # h_action[:, pos].assign(self._cast(prev_action))

            # if prev_state['t_stoch'] is not None:
            #     prev_stoch = tf.concat([self._cast(prev_state['t_stoch']), cur_stoch], 1)
            #     prev_action = tf.concat([self._cast(prev_state['t_action']), cur_action], 1)
            # else:
            #     prev_stoch = cur_stoch
            #     prev_action = cur_action

            # seq_len = tf.shape(prev_stoch)[1]
            # if seq_len > self._max_memory:
            #     prev_stoch = prev_stoch[:, 1:]
            #     prev_action = prev_action[:, 1:]
            _prev_stoch = tf.identity(h_stoch)
            _prev_action = tf.identity(h_action)

            prev_stoch = _prev_stoch
            prev_action = _prev_action

            # print(h_stoch[0].shape, h_action[1].shape)

            # print(h_stoch.shape, h_action.shape)
            # prev_stoch = tf.concat([h_stoch[:, pos + 1:], h_stoch[:, :pos + 1]], 1)
            # prev_action = tf.concat([h_action[:, pos + 1:], h_action[:, :pos + 1]], 1)
            # prev_stoch = tf.roll(h_stoch, )

        if self._discrete:
            shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
            prev_stoch = tf.reshape(prev_stoch, shape)

        x = tf.concat([prev_stoch, prev_action], -1)
        x = self.get('img_in', tfkl.Dense, self._hidden)(x)
        x = self.get('img_in_norm', NormLayer, self._norm)(x)
        x = self._act(x)

        if self.use_rnn:
            deter = prev_state['deter']
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        elif self.use_transformer:
            # x: [batch_size, seq_len, hidden]
            deter, _ = self._transformer(x, training=training)  # [batch_size, seq_len, hidden]
            x = deter = deter[:, -1, :]  # [batch_size, hidden]

        stats = self._suff_stats_ensemble(x)
        index = tf.random.uniform((), 0, self._ensemble, tf.int32)
        stats = {k: v[index] for k, v in stats.items()}
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()  # [batch_size, ]

        prior = {'stoch': stoch, 'deter': deter, **stats}
        if self.use_transformer:
            prior['t_stoch'] = _prev_stoch
            prior['t_action'] = _prev_action
            # prior['t_len'] = prev_state['t_len'] + 1

        return prior

    def _suff_stats_ensemble(self, inp):
        bs = list(inp.shape[:-1])
        inp = inp.reshape([-1, inp.shape[-1]])
        stats = []
        for k in range(self._ensemble):
            x = self.get(f'img_out_{k}', tfkl.Dense, self._hidden)(inp)
            x = self.get(f'img_out_norm_{k}', NormLayer, self._norm)(x)
            x = self._act(x)
            stats.append(self._suff_stats_layer(f'img_dist_{k}', x))
        stats = {
            k: tf.stack([x[k] for x in stats], 0)
            for k, v in stats[0].items()}
        stats = {
            k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
            for k, v in stats.items()}
        return stats

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
            logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
            return {'logit': logit}
        else:
            x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
            mean, std = tf.split(x, 2, -1)
            std = {
                'softplus': lambda: tf.nn.softplus(std),
                'sigmoid': lambda: tf.nn.sigmoid(std),
                'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {'mean': mean, 'std': std}

    def kl_loss(self, post, prior, forward, balance, free, free_avg):
        kld = tfd.kl_divergence
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)
        if balance == 0.5:
            value = kld(self.get_dist(lhs), self.get_dist(rhs))
            loss = tf.maximum(value, free).mean()
        else:
            value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
            value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
            if free_avg:
                loss_lhs = tf.maximum(value_lhs.mean(), free)
                loss_rhs = tf.maximum(value_rhs.mean(), free)
            else:
                loss_lhs = tf.maximum(value_lhs, free).mean()
                loss_rhs = tf.maximum(value_rhs, free).mean()
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        return loss, value


class Encoder(common.Module):

    def __init__(
            self, cnn_keys=r'image', mlp_keys=r'^$', act='elu', norm='none',
            cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
        self._cnn_keys = re.compile(cnn_keys)
        self._mlp_keys = re.compile(mlp_keys)
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers
        self._once = True

    @tf.function
    def __call__(self, obs):
        # print("in Encoder()", flush=True)
        outs = [self._cnn(obs), self._mlp(obs)]
        outs = [out for out in outs if out is not None]
        self._once = False
        # print("out Encoder()", flush=True)
        return tf.concat(outs, -1)

    def _cnn(self, obs):
        # print("in Encoder._cnn()", flush=True)
        inputs = {
            key: tf.reshape(obs[key], (-1,) + tuple(obs[key].shape[-3:]))
            for key in obs if self._cnn_keys.match(key)}
        if not inputs:
            # print("out Encoder._cnn()", flush=True)
            return None
        # print("1", flush=True)
        self._once and print('Encoder CNN inputs:', list(inputs.keys()))
        x = tf.concat(list(inputs.values()), -1)
        x = x.astype(prec.global_policy().compute_dtype)
        # print("2", flush=True)
        for i, kernel in enumerate(self._cnn_kernels):
            # print(f"3-{i} begin", flush=True)
            depth = 2 ** i * self._cnn_depth
            x = self.get(f'conv{i}', tfkl.Conv2D, depth, kernel, 2)(x)
            # print(f"3-{i} conv", flush=True)
            x = self.get(f'convnorm{i}', NormLayer, self._norm)(x)
            # print(f"3-{i} convnorm", flush=True)
            x = self._act(x)
            # print(f"3-{i} act", flush=True)
        # print("out Encoder._cnn()", flush=True)
        return x.reshape(list(obs['image'].shape[:-3]) + [-1])

    def _mlp(self, obs):
        # print("in Encoder._mlp()", flush=True)
        batch_dims = list(obs['reward'].shape)
        inputs = {
            key: tf.reshape(obs[key], [np.prod(batch_dims), -1])
            for key in obs if self._mlp_keys.match(key)}
        if not inputs:
            return None
        self._once and print('Encoder MLP inputs:', list(inputs.keys()))
        # print('\n'.join([str((k, v.shape, v.dtype)) for k, v in inputs.items()]))
        x = tf.concat(list(inputs.values()), -1)
        x = x.astype(prec.global_policy().compute_dtype)
        for i, width in enumerate(self._mlp_layers):
            x = self.get(f'dense{i}', tfkl.Dense, width)(x)
            x = self.get(f'densenorm{i}', NormLayer, self._norm)(x)
            x = self._act(x)
        # print("out Encoder._mlp()", flush=True)
        return x.reshape(batch_dims + [-1])


class Decoder(common.Module):

    def __init__(
            self, shapes, cnn_keys=r'image', mlp_keys=r'^$', act='elu', norm='none',
            cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
        self._shapes = shapes
        # print('decoder:', shapes)
        self._cnn_keys = re.compile(cnn_keys)
        self._mlp_keys = re.compile(mlp_keys)
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers
        self._once = True

    def __call__(self, features):
        features = tf.cast(features, prec.global_policy().compute_dtype)
        dists = {**self._cnn(features), **self._mlp(features)}
        self._once = False
        return dists

    def _cnn(self, features):
        shapes = {
            key: shape[-1] for key, shape in self._shapes.items()
            if self._cnn_keys.match(key)}
        if not shapes:
            return {}
        ConvT = tfkl.Conv2DTranspose
        x = self.get('convin', tfkl.Dense, 32 * self._cnn_depth)(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._cnn_depth])
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
            act, norm = self._act, self._norm
            if i == len(self._cnn_kernels) - 1:
                depth, act, norm = sum(shapes.values()), tf.identity, 'none'
            x = self.get(f'conv{i}', ConvT, depth, kernel, 2)(x)
            x = self.get(f'convnorm{i}', NormLayer, norm)(x)
            x = act(x)
        x = x.reshape(features.shape[:-1] + x.shape[1:])
        means = tf.split(x, list(shapes.values()), -1)
        dists = {
            key: tfd.Independent(tfd.Normal(mean, 1), 3)
            for (key, shape), mean in zip(shapes.items(), means)}
        self._once and print('Decoder CNN outputs:', list(dists.keys()))
        return dists

    def _mlp(self, features):
        shapes = {
            key: shape for key, shape in self._shapes.items()
            if self._mlp_keys.match(key)}
        if not shapes:
            return {}
        x = features
        for i, width in enumerate(self._mlp_layers):
            x = self.get(f'dense{i}', tfkl.Dense, width)(x)
            x = self.get(f'densenorm{i}', NormLayer, self._norm)(x)
            x = self._act(x)
        dists = {}
        for key, shape in shapes.items():
            dists[key] = self.get(f'dense_{key}', DistLayer, shape)(x)
        self._once and print('Decoder MLP outputs:', list(dists.keys()))
        return dists


class MLP(common.Module):

    def __init__(self, shape, layers, units, act='elu', norm='none', **out):
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._norm = norm
        self._act = get_act(act)
        self._out = out

    def __call__(self, features):
        x = tf.cast(features, prec.global_policy().compute_dtype)
        x = x.reshape([-1, x.shape[-1]])
        for index in range(self._layers):
            x = self.get(f'dense{index}', tfkl.Dense, self._units)(x)
            x = self.get(f'norm{index}', NormLayer, self._norm)(x)
            x = self._act(x)
        x = x.reshape(features.shape[:-1] + [x.shape[-1]])
        return self.get('out', DistLayer, self._shape, **self._out)(x)


class GRUCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, size, norm=False, act='tanh', update_bias=-1, **kwargs):
        super().__init__()
        self._size = size
        self._act = get_act(act)
        self._norm = norm
        self._update_bias = update_bias
        self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
        if norm:
            self._norm = tfkl.LayerNormalization(dtype=tf.float32)

    @property
    def state_size(self):
        return self._size

    @tf.function
    def call(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(tf.concat([inputs, state], -1))
        if self._norm:
            dtype = parts.dtype
            parts = tf.cast(parts, tf.float32)
            parts = self._norm(parts)
            parts = tf.cast(parts, dtype)
        reset, cand, update = tf.split(parts, 3, -1)
        reset = tf.nn.sigmoid(reset)
        cand = self._act(reset * cand)
        update = tf.nn.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class DistLayer(common.Module):

    def __init__(
            self, shape, dist='mse', min_std=0.1, init_std=0.0):
        self._shape = shape
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std

    def __call__(self, inputs):
        out = self.get('out', tfkl.Dense, np.prod(self._shape))(inputs)
        out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
        out = tf.cast(out, tf.float32)
        if self._dist in ('normal', 'tanh_normal', 'trunc_normal'):
            std = self.get('std', tfkl.Dense, np.prod(self._shape))(inputs)
            std = tf.reshape(std, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
            std = tf.cast(std, tf.float32)
        if self._dist == 'mse':
            dist = tfd.Normal(out, 1.0)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == 'normal':
            dist = tfd.Normal(out, std)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == 'binary':
            dist = tfd.Bernoulli(out)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == 'tanh_normal':
            mean = 5 * tf.tanh(out / 5)
            std = tf.nn.softplus(std + self._init_std) + self._min_std
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, common.TanhBijector())
            dist = tfd.Independent(dist, len(self._shape))
            return common.SampleDist(dist)
        if self._dist == 'trunc_normal':
            std = 2 * tf.nn.sigmoid((std + self._init_std) / 2) + self._min_std
            dist = common.TruncNormalDist(tf.tanh(out), std, -1, 1)
            return tfd.Independent(dist, 1)
        if self._dist == 'onehot':
            return common.OneHotDist(out)
        NotImplementedError(self._dist)


class NormLayer(common.Module):

    def __init__(self, name):
        if name == 'none':
            self._layer = None
        elif name == 'layer':
            self._layer = tfkl.LayerNormalization()
        else:
            raise NotImplementedError(name)

    def __call__(self, features):
        if not self._layer:
            return features
        return self._layer(features)


def get_act(name):
    if name == 'none':
        return tf.identity
    if name == 'mish':
        return lambda x: x * tf.math.tanh(tf.nn.softplus(x))
    elif hasattr(tf.nn, name):
        return getattr(tf.nn, name)
    elif hasattr(tf, name):
        return getattr(tf, name)
    else:
        raise NotImplementedError(name)
