import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import common


class EnsembleRSSM(common.Module):

    def __init__(
            self, config, ensemble=5, stoch=30, deter=200, hidden=200, discrete=False,
            act='elu', norm='none', std_act='softplus', min_std=0.1, exclude_deter_feat=False,
            use_transformer=False, num_actions=None, transformer=None, use_forward_loss=False,
            use_transformer_reward_head=False, encoder=None, inside_transformer=None):
        super().__init__()
        self.config = config
        self._ensemble = ensemble
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._act = get_act(act)
        self._norm = norm
        self._std_act = std_act
        self._min_std = min_std
        # self._seq_model = seq_model
        self._num_actions = num_actions
        self._exclude_deter_feat = exclude_deter_feat
        self.use_transformer = use_transformer
        self._use_forward_loss = use_forward_loss
        self.use_transformer_reward_head = use_transformer_reward_head and use_transformer
        # print("seq_model:", seq_model)

        # rnn state: {"stoch": [batch_size, stoch, discrete], "deter": [batch_size, deter]}
        # transformer state: {"stoch": [batch_size, stoch, discrete], "t_stoch": [batch_size, seq_len - 1, stoch, discrete], 
        #                     "t_action": [batch_size, seq_len - 1, n_action], "deter": [batch_size, deter]}
    
        self._cell = GRUCell(self._deter, norm=True)
        self._memory_size = transformer.pop("memory_size")
        self._inside_memory_size = inside_transformer.pop("memory_size")
        self._transformer_version = transformer.pop("version")
        self.transformer_num_layers = transformer["num_layers"]
        self._transformer_num_heads = transformer["num_heads"]
        self._transformer_params = transformer
        self._use_raw_input_in_transformer = config.use_raw_input_in_transformer and self.use_transformer
        self._use_independent_transformer = config.use_independent_transformer and self.use_transformer
        self._use_inside_transformer = config.use_inside_transformer and self._use_independent_transformer
        self.use_independent_transformer_encoder = config.use_independent_transformer_encoder
        self._include_transformer_embed = config.include_transformer_embed
        if self.use_transformer:
            if self._use_independent_transformer:
                assert self._use_raw_input_in_transformer
                transformer.pop('no_pe')
                transformer.pop('reverse_pe')
                self._transformer = common.TransformerNew(output_dim=hidden, **transformer)

                if self._use_inside_transformer:
                    self._inside_transformer = common.Transformer(output_dim=hidden, no_pe=True, **inside_transformer)
            else:
                self._transformer = common.Transformer(output_dim=hidden, **transformer)
        else:
            self._transformer = None
        self._transformer_encoder = None
        if self._use_raw_input_in_transformer:
            if (self._use_independent_transformer and self.use_independent_transformer_encoder) or (not self._use_independent_transformer):
                print("Transformer encoder:", encoder)
                self._transformer_encoder = encoder or common.Encoder(**config.encoder)
        self._importance_head = None
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    def set_importance_head(self, head):
        self._importance_head = head

    def transformer_encode(self, data, default=None):
        if self._transformer_encoder is None:
            return default
        else:
            return self._transformer_encoder(data)

    def initial(self, batch_size, transformer_weight=False):
        dtype = prec.global_policy().compute_dtype
        if self._discrete:
            stoch_size = [self._stoch, self._discrete]
            total_stoch = self._stoch * self._discrete
            state = dict(
                logit=tf.zeros([batch_size] + stoch_size, dtype),
                stoch=tf.zeros([batch_size] + stoch_size, dtype))
        else:
            stoch_size = [self._stoch]
            total_stoch = self._stoch
            state = dict(
                mean=tf.zeros([batch_size] + stoch_size, dtype),
                std=tf.zeros([batch_size] + stoch_size, dtype),
                stoch=tf.zeros([batch_size] + stoch_size, dtype))
        state['deter'] = self._cell.get_initial_state(None, batch_size, dtype)
        if self._use_forward_loss:
            state['forward_stoch'] = tf.zeros([batch_size] + stoch_size, dtype)
        if self.use_transformer:
            # state['deter'] = tf.zeros([batch_size, self._deter], dtype)
            if not self._use_independent_transformer:
                if self._use_raw_input_in_transformer:
                    # state['t_images'] = tf.zeros([batch_size, self._memory_size, *self.config.render_size, 3])
                    # state['t_actions'] = tf.zeros([batch_size, self._memory_size, self._num_actions])
                    state['t_hidden'] = tf.zeros([batch_size, self._memory_size, self._hidden])
                    state['t_counter'] = tf.zeros([batch_size, 1])
                else:
                    state['t_hidden'] = tf.zeros([batch_size, self._memory_size, total_stoch + self._deter])
                state['t_transformer'] = tf.zeros([batch_size, self._deter])
                # state['mask'] = tf.zeros([batch_size, 1, 1, self._memory_size])
                # state['weights'] = tf.zeros([batch_size, self._memory_size, self._memory_size])
                # state['x'] = tf.zeros([batch_size, self._deter])
                # state['t_stoch'] = tf.zeros([batch_size, self._memory_size] + stoch_size)
                # state['t_action'] = tf.zeros([batch_size, self._memory_size, self._num_actions])
                # state['t_len'] = tf.zeros([batch_size, 1], dtype=tf.int32)
                if transformer_weight:
                    for i in range(self.transformer_num_layers):
                        state[f't_weight_{i}'] = tf.zeros([batch_size, self._transformer_num_heads, self._memory_size])
                        # state[f't_weight_norm_{i}'] = tf.zeros([batch_size, self._transformer_num_heads])
            elif self._use_inside_transformer:
                state['t_memory'] = tf.zeros([batch_size, self._inside_memory_size, total_stoch + self._deter])
                state['t_importance'] = tf.zeros([batch_size, self._inside_memory_size])
        return state

    # @property
    # def use_rnn(self):
    #     return self._seq_model == 'rnn'

    # @property
    # def use_transformer(self):
    #     return self._seq_model == 'transformer'

    def calc_transformer_reward_hidden(self, embed, action, is_first, training, return_weight=False):
        print("in calc_transformer_reward_hidden()")
        print("embed.shape", embed.shape, flush=True)
        print("action.shape", action.shape, flush=True)
        print("is_first.shape", is_first.shape, flush=True)
        x = tf.concat([embed, action], -1)
        out, weight = self._transformer(x, is_first, training)
        print("out.shape", out.shape, flush=True)
        if return_weight:
            return out, weight
        else:
            return out

    @tf.function
    def observe(self, embed, transformer_embed, image, action, is_first, training, state=None, transformer_weight=False):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0], transformer_weight=transformer_weight)

        print("State:")
        for k, v in state.items():
            print(k, v.shape, flush=True)

        if self._use_independent_transformer:
            _state = dict()
            for k, v in state.items():
                if k != 't_transformer' and not k.startswith('t_weight_'):
                    _state[k] = v
            state = _state
            # try:
            #     state.pop('t_transformer')
            #     for i in range(self.transformer_num_layers):
            #         state.pop(f't_weight_{i}')
            # except KeyError:
            #     pass

        prev_image = tf.concat([tf.zeros_like(image)[:, :1], image[:, :-1]], 1)
        print("prev_image:", prev_image.shape, flush=True)
        print("is_first:", is_first.shape, flush=True)
        # print("image:", image.shape, flush=True)
        # print("action:", action.shape, flush=True)
        # print("embed:", embed.shape, flush=True)
        # print("is_first:", is_first.shape, flush=True)

        if self._use_independent_transformer:
            if self.use_independent_transformer_encoder:
                print("transformer_embed.shape", transformer_embed.shape, flush=True)
                _embed = transformer_embed
            else:
                print("embed.shape", embed.shape, flush=True)
                _embed = embed
                
            out, weight = self.calc_transformer_reward_hidden(_embed, action, is_first, training, return_weight=True)

            t_states = dict()
            t_states['t_transformer'] = out
            for i in range(self.transformer_num_layers):
                t_states[f't_weight_{i}'] = weight[f'dec{i}'].transpose([0, 2, 1, 3])
                print(f't_weight_{i}.shape', t_states[f't_weight_{i}'].shape, flush=True)  # [batch, length, head, length]

        post, prior = common.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs, training=training, transformer_weight=transformer_weight),
            (swap(prev_image), swap(action), swap(image), swap(embed), swap(transformer_embed), swap(is_first)), (state, state))

        print("Post:")
        for k, v in post.items():
            print(k, v.shape, flush=True)
        print("Prior:")
        for k, v in prior.items():
            print(k, v.shape, flush=True)

        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}

        if self._use_independent_transformer:
            post.update(t_states)

        print("Post:")
        for k, v in post.items():
            print(k, v.shape, flush=True)
        print("Prior:")
        for k, v in prior.items():
            print(k, v.shape, flush=True)

        return post, prior

    @tf.function
    def imagine(self, action, training, state=None, transformer_weight=False):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0])

        if self._use_independent_transformer:
            _state = dict()
            for k, v in state.items():
                if k != 't_transformer' and not k.startswith('t_weight_'):
                    _state[k] = v
            state = _state

        assert isinstance(state, dict), state
        action = swap(action)
        prior = common.static_scan(lambda prev, inputs: self.img_step(prev, inputs, training=training, transformer_weight=transformer_weight), action, state)
        # def _swap(k, v):
        #     print(k, v.shape)
        #     if k.startswith("t_"):
        #         return swap(v)
        #     else:
        #         return swap(v)

        print('in imagine')
        print('action.shape', action.shape)
        print('state')
        for k, v in state.items():
            print(k, v.shape, flush=True)
        print('prior')
        for k, v in prior.items():
            print(k, v.shape, flush=True)

        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = self._cast(state['stoch'])
        if self._discrete:
            shape = stoch.shape[:-2] + [self._stoch * self._discrete]
            stoch = tf.reshape(stoch, shape)
        if self._exclude_deter_feat:
            deter = tf.zeros_like(state['deter'])
        else:
            deter = state['deter']
        return tf.concat([stoch, deter], -1)

    def get_mask(self, name):
        def create_stoch_mask(_n):
            assert 0 <= _n <= self._stoch
            if self._discrete:
                z = self._discrete
            else:
                z = 1
            return tf.concat((tf.ones(_n * z), tf.zeros((self._stoch - _n) * z)), 0)
        if name == 'decoder':
            stoch_mask = create_stoch_mask(self._stoch // 2)
        elif name == 'reward' or name == 'discount' or name.startswith('int_reward'):
            stoch_mask = create_stoch_mask(self._stoch)
        elif name == 'transformer_reward':
            return self._cast(tf.ones(self._deter))
        else:
            raise NotImplementedError
        mask = tf.concat((stoch_mask, tf.ones(self._deter)), 0)
        return self._cast(mask)

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
    def obs_step(self, prev_state, prev_image, prev_action, cur_image, embed, t_embed, is_first, training, sample=True, transformer_weight=False):
        print("obs_step", transformer_weight)
        # if is_first.any():
        # print(list(prev_state.keys()), flush=True)
        # prev_state, prev_action = tf.nest.map_structure(
        #     lambda x: tf.einsum(
        #         'b,b...->b...', (1.0 - is_first.astype(tf.float32)).astype(x.dtype), x),
        #     (prev_state, prev_action))
        # mask = is_first.astype(tf.float32).expand
        print("is_first", is_first.shape)
        def _mask_fn(x):
            _mask = (1.0 - is_first.astype(tf.float32)).astype(x.dtype)
            _shape = list(is_first.shape) + [1] * (len(x.shape) - 1)
            return x * _mask.reshape(_shape)
        prev_state, prev_action = tf.nest.map_structure(
            _mask_fn, (prev_state, prev_action))
        # mask_fn = lambda x: tf.einsum('b,b...->b...', 1.0 - is_first.astype(x.dtype), x)
        # prev_action = mask_fn(prev_action)
        # for k in prev_state.keys():
        #     # if not k.startswith('t_'):
        #     prev_state[k] = mask_fn(prev_state[k])
        prior = self.img_step(prev_state, prev_action, training, sample=sample, transformer_weight=transformer_weight)

        use_transformer = self.use_transformer and self._use_raw_input_in_transformer and not self._use_independent_transformer
        if use_transformer:
            # h_images = self._cast(tf.identity(prev_state['t_images']))
            # h_actions = self._cast(tf.identity(prev_state['t_actions']))
            # h_images = tf.stop_gradient(tf.concat([h_images[:, 1:], prev_image[:, tf.newaxis]], 1))
            # h_actions = tf.stop_gradient(tf.concat([h_actions[:, 1:], prev_action[:, tf.newaxis]], 1))
            # mask = 1. - self._cast(tf.math.equal(tf.reduce_sum(h_images ** 2, [-3, -2, -1]), 0)[:, :, tf.newaxis])  # [batch, mem, 1]
            # h_images_embed = self._transformer_encoder({"image": h_images})
            # h_out = tf.concat([h_images_embed, h_actions], -1)
            # h_out = self.get('h_out', tfkl.Dense, self._hidden)(h_out)
            # h_out = self.get('h_out_norm', NormLayer, self._norm)(h_out)
            # h_out = h_out * mask

            print("prev_image shape:", prev_image.shape)
            print("cur_image shape:", cur_image.shape)
            mask = tf.stop_gradient(1. - self._cast(tf.math.equal(tf.reduce_sum(prev_image ** 2, [-3, -2, -1]), 0.)))  # [batch]

            prev_image_embed = self._transformer_encoder({"image": prev_image})
            print("prev_image_embed shape:", prev_image_embed.shape)
            print("prev_action shape:", prev_action.shape)
            # prev_out = tf.concat([prev_image_embed, prev_action, prev_state['t_counter']], -1)
            prev_out = tf.concat([prev_image_embed, prev_action], -1)
            # prev_out = prev_image_embed
            prev_out = self.get('prev_out', tfkl.Dense, self._hidden)(prev_out)
            prev_out = self.get('prev_out_norm', NormLayer, self._norm)(prev_out)
            print("prev_out shape:", prev_out.shape)

            prev_out = prev_out * mask[:, tf.newaxis]
            print("prev_out shape:", prev_out.shape)

            h_hidden = tf.identity(prev_state['t_hidden'])
            h_hidden = tf.concat([h_hidden[:, 1:, :], prev_out[:, tf.newaxis, :]], 1)

            cur_out = tf.concat([embed, prev_action], -1)
            cur_out = self.get('cur_out', tfkl.Dense, self._hidden)(cur_out)
            cur_out = self.get('cur_out_norm', NormLayer, self._norm)(cur_out)

            transformer_out, weights, weights_norm, mask = self._transformer(h_hidden, cur_out, training=training)
            print('transformer_out shape:', transformer_out.shape)

            transformer_out = tf.concat([transformer_out, cur_out], -1)
            transformer_out = self.get("transformer_out", tfkl.Dense, self._hidden)(transformer_out)
            transformer_out = self.get("transformer_out_norm", NormLayer, self._norm)(transformer_out)

            # x = tf.concat([prior['deter'], embed, tf.stop_gradient(self._transformer_encoder({"image": cur_image}))], -1)
            if self._include_transformer_embed:
                x = tf.concat([prior['deter'], embed, tf.stop_gradient(self._transformer_encoder({"image": cur_image}))], -1)
            else:
                x = tf.concat([prior['deter'], embed], -1)
            x = self.get('t_obs_out', tfkl.Dense, self._hidden)(x)
            x = self.get('t_obs_out_norm', NormLayer, self._norm)(x)
            x = self._act(x)
        else:
            if self._include_transformer_embed:
                x = tf.concat([prior['deter'], embed, tf.stop_gradient(t_embed)], -1)
            else:
                x = tf.concat([prior['deter'], embed], -1)
            x = self.get('obs_out', tfkl.Dense, self._hidden)(x)
            x = self.get('obs_out_norm', NormLayer, self._norm)(x)
            x = self._act(x)
        
        stats = self._suff_stats_layer('obs_dist', x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
        if self.use_transformer and not self._use_independent_transformer:
            if self._use_raw_input_in_transformer:
                post['t_hidden'] = tf.identity(h_hidden)
                # post['t_images'] = tf.stop_gradient(tf.identity(h_images))
                # post['t_actions'] = tf.stop_gradient(tf.identity(h_actions))
                post['t_transformer'] = tf.identity(transformer_out)
                post['t_counter'] = tf.stop_gradient(prev_state['t_counter'] + 1)
            else:
                post['t_hidden'] = tf.stop_gradient(tf.identity(prior['t_hidden']))
                post['t_transformer'] = tf.identity(prior['t_transformer'])
            if transformer_weight:
                for i in range(self.transformer_num_layers):
                    if self._use_raw_input_in_transformer:
                        post[f't_weight_{i}'] = tf.stop_gradient(weights[f'dec{i}'])
                        post[f't_weight_norm_{i}'] = tf.stop_gradient(weights_norm[f'dec{i}'])
                    else:
                        post[f't_weight_{i}'] = tf.stop_gradient(tf.identity(prior[f't_weight_{i}']))
                        post[f't_weight_norm_{i}'] = tf.stop_gradient(tf.identity(prior[f't_weight_norm_{i}']))
                    print(f"post['t_weight_{i}']", post[f't_weight_{i}'].shape)
            # post['mask'] = tf.stop_gradient(tf.identity(prior['mask']))
            # post['weights'] = tf.stop_gradient(tf.identity(prior['weights']))
            # post['x'] = tf.stop_gradient(tf.identity(prior['x']))
            # post['t_stoch'] = tf.identity(prior['t_stoch'])
            # post['t_action'] = tf.identity(prior['t_action'])
            # post['t_len'] = tf.identity(prior['t_len'])
        return post, prior

    @tf.function
    def img_step(self, prev_state, prev_action, training, sample=True, transformer_weight=False):
        print("img_step", transformer_weight)
        _prev_stoch = None
        _prev_action = None
        prev_stoch = self._cast(prev_state['stoch'])
        prev_action = self._cast(prev_action)
        # if self.use_transformer:
            # cur_stoch = tf.expand_dims(self._cast(prev_state['stoch']), 1)
            # cur_action = tf.expand_dims(self._cast(prev_action), 1)

            # pos = prev_state['t_len'] % self._memory_size  # [batch, 1]
            # print(prev_state['t_len'], type(prev_state['t_stoch']).__name__)
            # h_stoch = self._cast(tf.identity(prev_state['t_stoch']))
            # h_action = self._cast(tf.identity(prev_state['t_action']))

            # h_stoch = tf.roll(h_stoch, shift=-1, axis=1)  # [batch, memory, stoch]
            # h_action = tf.roll(h_action, shift=-1, axis=1)
            
            # h_hidden = self._cast(tf.identity(prev_state['t_hidden']))
            # h_hidden = tf.roll(h_hidden, shift=-1, axis=1)

            # bs = tf.shape(h_stoch)[0]
            # sl = tf.shape(h_stoch)[1]
            # indices = tf.stack([tf.range(bs), tf.ones(bs, dtype=tf.int32) * (sl - 1)], 1)
            # print(indices)
            # h_stoch = tf.tensor_scatter_nd_update(h_stoch, indices, self._cast(tf.stop_gradient(prev_state['stoch'])))
            # h_action = tf.tensor_scatter_nd_update(h_action, indices, self._cast(tf.stop_gradient(prev_action)))
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
            # _prev_stoch = self._cast(tf.identity(h_stoch))
            # _prev_action = self._cast(tf.identity(h_action))

            # prev_stoch = _prev_stoch
            # prev_action = _prev_action

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

        if self._use_forward_loss:
            forward_x = tf.concat([prev_stoch, prev_action], -1)
            forward_x = self.get('forward_img_in', tfkl.Dense, self._hidden)(forward_x)
            forward_x = self.get('forward_img_in_norm', NormLayer, self._norm)(forward_x)
            forward_x = self._act(forward_x)

        use_transformer = self.use_transformer and (not self._use_raw_input_in_transformer or self._use_inside_transformer)
        print('use_transformer in img?', use_transformer, flush=True)

        if use_transformer:
            if self._use_inside_transformer:
                t_memory = self._cast(tf.identity(prev_state['t_memory']))
                t_importance = self._cast(tf.identity(prev_state['t_importance']))
                last_token = tf.stop_gradient(tf.concat([prev_stoch, prev_state['deter']], 1))
                last_importance = tf.stop_gradient(self._importance_head(last_token).mode())
                t_memory = tf.concat((t_memory, last_token[:, tf.newaxis, :]), 1)
                t_importance = tf.concat((t_importance, last_importance[:, tf.newaxis]), 1)
                _, indices = tf.math.top_k(t_importance, k=self._inside_memory_size)
                t_memory = tf.stop_gradient(tf.gather(t_memory, indices, batch_dims=-1))
                t_importance = tf.stop_gradient(tf.gather(t_importance, indices, batch_dims=-1))
                transformer_out, weights, _ = self._inside_transformer(t_memory, x, training=training)
            else:
                h_hidden = self._cast(tf.identity(prev_state['t_hidden']))
                last_token = tf.stop_gradient(tf.concat([prev_stoch, prev_state['deter']], 1))
                h_hidden = tf.stop_gradient(tf.concat([h_hidden[:, 1:, :], last_token[:, tf.newaxis, :]], 1))
                transformer_out, weights, mask = self._transformer(h_hidden, x, training=training)
                transformer_out_copy = tf.identity(transformer_out)

        if self._transformer_version == 1:
            if use_transformer:
                x = self.get('transformer_out_ln', tfkl.LayerNormalization)(x + transformer_out)
            deter = prev_state['deter']
            x, deter = self._cell(x, [deter])
            deter = deter[0]
        elif self._transformer_version == 2:
            if use_transformer:
                x = deter = self.get('transformer_out_ln', tfkl.LayerNormalization)(x + transformer_out)
            else:
                deter = prev_state['deter']
                x, deter = self._cell(x, [deter])
                deter = deter[0]  # Keras wraps the state in a list.
        elif self._transformer_version == 3:
            if use_transformer:
                transformer_out = self.get('transformer_out_ln', tfkl.LayerNormalization)(x + transformer_out)
                deter = prev_state['deter']
                x, deter = self._cell(transformer_out, [deter])
                deter = deter[0]
                x = self.get('transformer_out_ln_2', tfkl.LayerNormalization)(x + transformer_out)
            else:
                deter = prev_state['deter']
                x, deter = self._cell(x, [deter])
                deter = deter[0]
        # elif self.use_transformer:
        #     h_hidden = self._cast(tf.identity(prev_state['t_hidden']))
        #     # h_hidden = tf.stop_gradient(tf.roll(h_hidden, shift=-1, axis=1))
        #     # bs = tf.shape(h_hidden)[0]
        #     # sl = tf.shape(h_hidden)[1]
        #     # indices = tf.stack([tf.range(bs), tf.ones(bs, dtype=tf.int32) * (sl - 1)], 1)
        #     # print(indices)
        #     # h_hidden = tf.stop_gradient(tf.tensor_scatter_nd_update(h_hidden, indices, x))
        #     _x = tf.identity(x)
        #     # h_hidden = tf.stop_gradient(tf.concat([h_hidden[:, 1:, :], tf.stop_gradient(prev_stoch)[:, tf.newaxis, :]], 1))
        #     h_hidden = tf.stop_gradient(tf.concat([h_hidden[:, 1:, :], tf.stop_gradient(prev_stoch['deter'])[:, tf.newaxis, :]], 1))
        #     out, weights, mask = self._transformer(h_hidden, training=training)  # [batch_size, seq_len, hidden]
        #     out = tf.concat([out[:, -1, :], _x], -1)
        #     out = self.get('transformer_out', tfkl.Dense, self._hidden)(out)
        #     x = deter = self.get('transformer_out_norm', NormLayer, self._norm)(out)
        #     # x = deter = h_hidden[:, -1, :]  # [batch_size, hidden]

        stats = self._suff_stats_ensemble(x)
        index = tf.random.uniform((), 0, self._ensemble, tf.int32)
        stats = {k: v[index] for k, v in stats.items()}
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()  # [batch_size, ]

        prior = {'stoch': stoch, 'deter': deter, **stats}

        if self._use_forward_loss:
            forward_stats = self._suff_stats_ensemble(forward_x)
            forward_index = tf.random.uniform((), 0, self._ensemble, tf.int32)
            forward_stats = {k: v[index] for k, v in forward_stats.items()}
            forward_dist = self.get_dist(forward_stats)
            forward_stoch = forward_dist.sample() if sample else forward_dist.mode()

            prior['forward_stoch'] = forward_stoch

        if use_transformer:
            if not self._use_inside_transformer:
                prior['t_hidden'] = tf.stop_gradient(h_hidden)
                prior['t_transformer'] = tf.identity(transformer_out_copy)
                print("transformer_out", transformer_out_copy.shape, transformer_out_copy)
                if transformer_weight:
                    for i in range(self.transformer_num_layers):
                        prior[f't_weight_{i}'] = tf.stop_gradient(weights[f'dec{i}'])
                        print(f"prior['t_weight_{i}']", prior[f't_weight_{i}'].shape)
            else:
                prior['t_memory'] = tf.stop_gradient(t_memory)
                prior['t_importance'] = tf.stop_gradient(t_importance)
            # prior['weights'] = tf.stop_gradient(weights['enc0'])
            # prior['mask'] = tf.stop_gradient(mask)
            # prior['x'] = tf.stop_gradient(_x)
            # prior['t_stoch'] = tf.stop_gradient(_prev_stoch)
            # prior['t_action'] = tf.stop_gradient(_prev_action)
            # prior['t_len'] = prev_state['t_len'] + 1
        # elif self.use_transformer:
        #     for k, v in prev_state.items():
        #         if k.startswith("t_"):
        #             prior[k] = tf.zeros_like(v)

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
            print(f"encoder {i}-shape:", x.shape)
        # print("out Encoder._cnn()", flush=True)
        return x.reshape(list(obs['image'].shape[:-3]) + [-1])

    def _mlp(self, obs):
        print("in Encoder._mlp()", flush=True)
        print(obs.keys())
        valid_inputs = [key for key in obs if self._mlp_keys.match(key)]
        if not valid_inputs:
            return None
        batch_dims = list(obs['reward'].shape)
        inputs = {
            key: tf.reshape(obs[key], [np.prod(batch_dims), -1])
            for key in obs if self._mlp_keys.match(key)}
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
            print(f"decoder {i}-shape:", x.shape)
        x = x.reshape(features.shape[:-1] + x.shape[1:])
        print("decoder shape:", shapes, x.shape)
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
        # print(shape, dist, flush=True)
        self._shape = shape
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std

    def __call__(self, inputs):
        out = self.get('out', tfkl.Dense, np.prod(self._shape))(inputs)
        out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
        out = tf.cast(out, tf.float32)
        # print("DistLayer.__call__", self._dist, flush=True)
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
