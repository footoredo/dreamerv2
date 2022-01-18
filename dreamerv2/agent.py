import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import common
import expl


class Agent(common.Module):

    def __init__(self, config, logger, step, shapes):
        self.config = config
        self._logger = logger
        self._num_act = shapes['action'][-1]
        self._counter = step
        self.step = tf.Variable(int(self._counter), tf.int64)
        self.wm = WorldModel(self.step, shapes, self._num_act, config)

        # print("wm:", self.wm.variables)
        self._task_behavior = ActorCritic(config, self.step, self._num_act)
        if config.expl_behavior == 'greedy':
            self._expl_behavior = self._task_behavior
        else:
            reward = lambda seq: self.wm.heads['reward'](seq['feat']).mode()
            inputs = config, self.wm, self._num_act, self.step, reward
            self._expl_behavior = getattr(expl, config.expl_behavior)(*inputs)

    @tf.function
    def policy(self, obs, state=None, mode='train'):
        obs = tf.nest.map_structure(tf.tensor, obs)
        tf.py_function(lambda: self.step.assign(
            int(self._counter), read_value=False), [], [])
        processed_obs = self.wm.preprocess(obs)
        if state is None:
            latent = self.wm.rssm.initial(len(obs['reward']))
            prev_image = tf.zeros_like(processed_obs['image'])
            action = tf.zeros((len(obs['reward']), self._num_act))
            state = latent, prev_image, action
        latent, prev_image, action = state
        embed = self.wm.encoder(processed_obs)
        t_embed = self.wm.rssm.transformer_encode(processed_obs, tf.zeros_like(embed))
        sample = (mode == 'train') or not self.config.eval_state_mean
        latent, _ = self.wm.rssm.obs_step(
            latent, prev_image, action, processed_obs['image'], embed, t_embed, obs['is_first'], sample)
        feat = self.wm.rssm.get_feat(latent)
        if mode == 'eval':
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
            noise = self.config.eval_noise
        elif mode == 'explore':
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
            noise = self.config.expl_noise
        elif mode == 'train':
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
            noise = self.config.expl_noise
        elif mode == 'random':
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
            noise = 1.0
        action = common.action_noise(action, noise, self.config.discrete)

        # mat = np.eye(17)
        # mat[7, 7] = 0
        # mat[7, 0] = 1  # replace place_stone to 
        # mat = tf.constant(mat, dtype=action.dtype)
        # action = tf.matmul(action, mat)
        
        outputs = {'action': action}
        state = (latent, processed_obs['image'], action)
        return outputs, state

    @tf.function(experimental_compile=False)
    def train(self, data, state=None):
        print("in agent train()", flush=True)
        metrics = {}
        # for k, v in data.items():
        #     print(k, type(v))
        state, outputs, mets, model_loss, prior = self.wm.train(data, state)
        metrics.update(mets)
        start = outputs['post']
        if self.config.make_graph:
            return model_loss
        # print("self.wm.train states:", flush=True)
        # for k, v in start.items():
        #     p = 1
        #     for d in v.shape:
        #         p *= d
        #     print(k, v.shape, p)
        # for k, v in start.items():
        #     print(k, v.device)
        # return model_loss, prior
        def reward_fn(seq):
            if self.config.use_transformer_reward:
                print("seq['feat'].shape", seq['feat'].shape, flush=True)  # [length, batch, ...]
                dists = self.wm.heads['decoder'](seq['feat'])

                imagined_obs = dict()
                for key, dist in dists.items():
                    imagined_obs[key] = dist.mode()

                if self.wm.rssm.use_independent_transformer_encoder:
                    _embed = self.wm.rssm.transformer_encode(imagined_obs)
                else:
                    _embed = self.wm.encoder(imagined_obs)

                swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
                
                out = self.wm.rssm.calc_independent_transformer_hidden(swap(_embed), swap(seq['action']), 
                    swap(tf.zeros_like(seq['action'])[:, :, 0]), training=True, return_weight=False)
                r = swap(self.wm.heads['transformer_reward'](out).mode())
            else:
                r = self.wm.heads['reward'](seq['feat']).mode()
            if self.config.use_int_reward:
                for source in self.config.int_reward_sources:
                    coef = self.config.int_reward_coef.get(source, 1.0)
                    print(f'int reward source: {source} coef: {coef}')
                    r += coef * self.wm.heads[f'int_reward_{source}'](seq['feat']).mode()
            return r
        if not self.config.no_behavior_training:
            metrics.update(self._task_behavior.train(
                self.wm, start, data['is_terminal'], reward_fn))
            if self.config.expl_behavior != 'greedy':
                mets = self._expl_behavior.train(start, outputs, data)[-1]
                metrics.update({'expl_' + key: value for key, value in mets.items()})
        print("out agent train()", flush=True)
        return state, metrics

    @tf.function(experimental_compile=False)
    def report(self, data, return_data=False):
        print(f"in report(), return_data={return_data}")
        report = {}
        data = self.wm.preprocess(data)
        for k, v in data.items():
            print(k, v.shape, flush=True)
        rtn = None
        for key in data:
            if re.match(self.config.decoder.cnn_keys, key):
                name = key.replace('/', '_')
                pred_dict = self.wm.video_pred(data, key, self._task_behavior if not self.config.no_behavior_training else None)
                report[f'openl_{name}'] = pred_dict['images']
                if return_data:
                    # print("save pred data")
                    rtn = pred_dict

        if return_data:  
            return report, rtn
        else:
            return report


class WorldModel(common.Module):

    def __init__(self, step, shapes, num_actions, config):
        self.step = step
        self.config = config
        self.encoder = common.Encoder(**config.encoder)
        self.rssm = common.EnsembleRSSM(config, encoder=None, num_actions=num_actions, **config.rssm)

        self.heads = {}

        def add_head(_name, _module, *args, **kwargs):
            if config.use_head_mask:
                self.heads[_name] = common.MaskLayer(lambda: _module(*args, **kwargs), self.rssm.get_mask(_name), gradient_mask=config.head_mask.gradient)
            else:
                self.heads[_name] = _module(*args, **kwargs)

        add_head('decoder', common.Decoder, shapes, **config.decoder)
        # self.heads['decoder'] = common.Decoder(shapes, **config.decoder)
        add_head('reward', common.MLP, [], **config.reward_head)
        # self.heads['reward'] = common.MLP([], **config.reward_head)
        self._use_transformer_reward_head = config.rssm.use_transformer_reward_head and config.rssm.use_transformer
        if self._use_transformer_reward_head:
            add_head('transformer_reward', common.MLP, [], **config.reward_head)
            # self.heads['transformer_reward'] = common.MLP([], **config.reward_head)
        self._myopic_prediction = config.myopic_prediction
        if self._myopic_prediction:
            add_head('myopic_reward', common.MLP, [], **config.reward_head)
        self._use_int_reward = config.use_int_reward
        if self._use_int_reward:
            print("use int reward!", flush=True)
            self._int_reward_sources = config.int_reward_sources
            for source in self._int_reward_sources:
                add_head(f'int_reward_{source}', common.MLP, [], **config.reward_head)
                if source == 'attention':
                    self.rssm.set_importance_head(self.heads['int_reward_attention'])
            # self.heads['int_reward'] = common.MLP([], **config.reward_head)
        # print("wm.reward:", self.heads['reward'].variables)
        # self._use_attention_int_reward = config.use_attention_int_reward
        if config.pred_discount:
            add_head('discount', common.MLP, [], **config.discount_head)
            # self.heads['discount'] = common.MLP([], **config.discount_head)
        for name in config.grad_heads:
            assert name in self.heads, name
        self.model_opt = common.Optimizer('model', **config.model_opt)
        self._bootstrap_frames = config.bootstrap_frames
        self._video_pred_batches = config.video_pred_batches
        # self._running_stats = {}

    def train(self, data, state=None):
        print("in wm train()", flush=True)
        with tf.GradientTape() as model_tape:
            model_loss, state, outputs, metrics, prior = self.loss(data, state)
        print("model_loss", model_loss, flush=True)
        print(metrics.keys(), flush=True)
        for k, v in metrics.items():
            print(k, v, flush=True)
        # print("1", flush=True)
        modules = [self.encoder, self.rssm, *self.heads.values()]
        # print("2", flush=True)
        metrics.update(self.model_opt(model_tape, model_loss, modules))
        print("out wm train()", flush=True)
        return state, outputs, metrics, model_loss, prior

    def calc_t_importance(self, t_weight, truth_reward, pred_reward, t_pred_reward, myopic_pred_reward=None, source=None, reduction=None):
        print("in calc_t_importance")
        print("t_weight.shape", t_weight.shape)
        print("truth_reward.shape", truth_reward.shape)
        print("pred_reward.shape", pred_reward.shape)
        print("t_pred_reward.shape", t_pred_reward.shape)

        if source is None:
            source = self.config.future_importance_source

        if reduction is None:
            reduction = self.config.future_importance_reduction

        t_weight = tf.identity(t_weight)  # [batch, length, num_heads, length], logits
        t_weight = tf.nn.softmax(t_weight, axis=-1)  # [batch, length, num_heads, length], weight
        if reduction == 'mean':
            t_weight = tf.reduce_mean(t_weight, -2)  # [batch, length, length]
        elif reduction == 'max':
            t_weight = tf.reduce_max(t_weight, -2)  # [batch, length, length]
        else:
            raise NotImplementedError
        identity = tf.eye(t_weight.shape[1])   # [length, length]
        identity = tf.expand_dims(identity, 0) # [1, length, length]
        t_weight = tf.multiply(1 - identity, t_weight)  # only cares attention in the past steps
        
        if source == 'reward':
            item = truth_reward
        elif source == 'abs_reward':
            item = tf.abs(truth_reward)
        elif source == 'reward_diff':
            item = tf.abs(pred_reward - t_pred_reward)
        elif source == 'reward_diff_myopic':
            item = tf.abs(myopic_pred_reward - t_pred_reward)
        else:
            raise NotImplementedError

        t_importance = tf.multiply(tf.expand_dims(item, -1), t_weight)  # [batch, length, length] -> pairwise importance

        return t_importance
        

    def loss(self, data, state=None):
        # print("in loss()", flush=True)
        data = self.preprocess(data)
        # print("1", flush=True)
        embed = self.encoder(data)
        transformer_embed = self.rssm.transformer_encode(data, tf.zeros_like(embed))
        # print("2", flush=True)
        post, prior, state_transformer_stats = self.rssm.observe(
            embed, transformer_embed, data['image'], data['action'], data['is_first'], training=True, state=state, transformer_weight=True)
        # print("3", flush=True)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        assert len(kl_loss.shape) == 0
        likes = {}
        losses = {'kl': kl_loss}
        if state_transformer_stats is not None:
            state_transformer_kl_loss, state_transformer_kl_value = self.rssm.kl_loss(post, state_transformer_stats, forward=False, balance=1.0, free=0.0, free_avg=True)
            losses["state_transformer_kl"] = state_transformer_kl_loss
        feat = self.rssm.get_feat(post)
        # print(feat.shape)

        myopic_pred_reward = None
        
        for name, head in self.heads.items():
            if name.startswith("int_reward"): 
                continue
            if name == "transformer_reward":
                # print("transformer_reward in loss", post['t_transformer'].shape, post['t_transformer'])
                dist = head(post['t_transformer'])
                t_pred_reward = tf.stop_gradient(dist.mode())
                like = tf.cast(dist.log_prob(data["reward"]), tf.float32)
                losses["transformer_reward"] = -like.mean()
            elif name == 'myopic_reward':
                dist = head(post['myopic_out'])
                myopic_pred_reward = tf.stop_gradient(dist.mode())
                like = tf.cast(dist.log_prob(data["reward"]), tf.float32)
                losses["myopic_reward"] = -like.mean()
            else:
                grad_head = (name in self.config.grad_heads)
                inp = feat if grad_head else tf.stop_gradient(feat)
                out = head(inp)
                dists = out if isinstance(out, dict) else {name: out}
                for key, dist in dists.items():
                    if key == 'reward':
                        pred_reward = tf.stop_gradient(dist.mode())
                    # print(key, data[key].shape, dist.mean().shape)
                    like = tf.cast(dist.log_prob(data[key]), tf.float32)
                    # if not key in self._running_stats:
                    #     self._running_stats[key] = common.RunningStats(like.shape)
                    # self._running_stats[key].push(-like)
                    likes[key] = like
                    losses[key] = -like.mean()
        
        metrics = {}

        if 't_weight_0' in post:
            # t_weight = tf.identity(post[f't_weight_{self.rssm.transformer_num_layers - 1}'])  # [batch, length, num_heads, length], logits
            # t_weight = tf.nn.softmax(t_weight, axis=-1)  # [batch, length, num_heads, length], weight
            # t_weight = tf.reduce_mean(t_weight, -2)  # [batch, length, length]
            # identity = tf.eye(t_weight.shape[1])   # [length, length]
            # identity = tf.expand_dims(identity, 0) # [1, length, length]
            # t_weight = tf.multiply(1 - identity, t_weight)  # only cares attention in the past steps
            # if self.config.future_importance_source == 'reward':
            #     source = data['reward']
            # elif self.config.future_importance_source == 'abs_reward':
            #     source = tf.abs(data['reward'])
            # elif self.config.future_importance_source == 'reward_diff':
            #     source = tf.abs(pred_reward - t_pred_reward)
            # else:
            #     raise NotImplementedError
            # t_importance = tf.multiply(tf.expand_dims(source, -1), t_weight)  # [batch, length, length] -> pairwise importance

            t_importance = self.calc_t_importance(post[f't_weight_{self.rssm._transformer.num_layers - 1}'], data['reward'], pred_reward, t_pred_reward, myopic_pred_reward)

        if self._use_int_reward:

            def _add_int_reward_loss(_source, _int_reward):
                key = f'int_reward_{_source}'
                inp = tf.stop_gradient(feat)
                dist = self.heads[key](inp)
                like = tf.cast(dist.log_prob(_int_reward), tf.float32)
                losses[key] = -like.mean()
                metrics[f'{key}_max'] = _int_reward.max()
                metrics[f'{key}_min'] = _int_reward.min()
                metrics[f'{key}_mean'] = _int_reward.mean()
                metrics[f'{key}_std'] = _int_reward.std()

            if 'expl' in self._int_reward_sources:
                model_like = 0
                print("in calc int from expl")
                for k, v in likes.items():
                    _v = (v.mean() - v) / (v.std() + 1e-8)
                    mask = tf.cast(_v > 1, tf.float32)
                    _v = mask * _v  # only keep significant reward (> 1 std)
                    print(k, _v.shape)
                    metrics[f'{k}_like_max'] = _v.max()
                    metrics[f'{k}_like_min'] = _v.min()
                    metrics[f'{k}_like_std'] = _v.std()
                    model_like += self.config.int_reward_scales.get(k, 0.0) * _v

                _add_int_reward_loss('expl', model_like)

            if 'attention' in self._int_reward_sources:
                t_int_reward = tf.reduce_sum(t_importance, -2)  # [batch, length]

                _add_int_reward_loss('attention', tf.stop_gradient(t_int_reward))

        # if self._use_int_reward and 'expl' in self._int_reward_sources:
        #     model_like = 0
        #     print("in calc int from expl")
        #     for k, v in likes.items():
        #         _v = (v.mean() - v) / (v.std() + 1e-8)
        #         mask = tf.cast(_v > 1, tf.float32)
        #         _v = mask * _v  # only keep significant reward (> 1 std)
        #         print(k, _v.shape)
        #         metrics[f'{k}_like_max'] = _v.max()
        #         metrics[f'{k}_like_min'] = _v.min()
        #         metrics[f'{k}_like_std'] = _v.std()
        #         model_like += self.config.int_reward_scales.get(k, 0.0) * _v
        #     int_reward = model_like
        #     # data["int_reward"] = int_reward
        #     inp = tf.stop_gradient(feat)
        #     # inp = feat
        #     dist = self.heads["int_reward_expl"](inp)
        #     like = tf.cast(dist.log_prob(int_reward), tf.float32)
        #     # print("model_like", model_like.shape, inp.shape, like.shape, flush=True)
        #     likes["int_reward_expl"] = like
        #     losses["int_reward_expl"] = -like.mean()
        #     metrics['int_reward_expl_max'] = int_reward.max()
        #     metrics['int_reward_expl_min'] = int_reward.min()
        #     metrics['int_reward_expl_mean'] = int_reward.mean()
        #     metrics['int_reward_expl_std'] = int_reward.std()

        # if self.rssm.use_transformer:
        #     losses['transformer_weight_norm'] = 0
        #     for i in range(self.rssm.transformer_num_layers):
        #         losses['transformer_weight_norm'] += post[f't_weight_norm_{i}'].mean()
            
        model_loss = 0
        # if self._use_int_reward:
        #     model_loss += losses["int_reward"]
        # print("losses:", flush=True)
        # model_loss = tf.zeros([], dtype=tf.float32)
        for k, v in losses.items():
            # print(k, v.shape, v, self.config.loss_scales.get(k, 1.0), flush=True)
            model_loss += self.config.loss_scales.get(k, 1.0) * v
            
        # model_loss = sum(
        #     self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
        outs = dict(
            embed=embed, feat=feat, post=post,
            prior=prior, likes=likes, kl=kl_value)
        metrics.update({f'{name}_loss': value for name, value in losses.items()})
        metrics['model_kl'] = kl_value.mean()
        if state_transformer_stats is not None:
            metrics['state_transformer_kl_value'] = state_transformer_kl_value.mean()
        metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
        metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
        # if self.rssm.use_transformer:
        #     for i in range(self.rssm.transformer_num_layers):
        #         metrics[f'transformer_weight_norm_{i}'] = tf.sqrt(post[f't_weight_norm_{i}']).mean()

        # def get_last(k, v):
        #     if k.startswith('t_'):
        #         return v[-1]
        #     else:
        #         return v[:, -1]
        last_state = {k: v[:, -1] for k, v in post.items()}
        # print("out loss()", flush=True)
        return model_loss, last_state, outs, metrics, prior

    def imagine(self, policy, start, is_terminal, horizon):
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        # for k, v in start.items():
        #     print(k, type(v))
        #     if type(v) == list:
        #         print(len(v), v[0].shape)
        #     else:
        #         print(v.shape)
        # print("in imagine")
        # for k, v in start.items():
        #     print(k, v.device)
        # def _flatten(k, v):
        #     if k.startswith('t_'):
        start = {k: flatten(v) for k, v in start.items()}
        start['feat'] = self.rssm.get_feat(start)
        start['action'] = tf.zeros_like(policy(start['feat']).mode())
        # print("in imagine:", start.keys())  # dict_keys(['logit', 'stoch', 'deter', 'feat', 'action'])
        # print("in imagine:", start.items())
        seq = {k: [v[:]] for k, v in start.items() if not k.startswith('t_')}
        t_states = {k: v[:] for k, v in start.items() if k.startswith('t_')}
        for h in range(horizon):
            action = policy(tf.stop_gradient(seq['feat'][-1])).sample()
            states = {k: v[-1][:] for k, v in seq.items()}
            states.update({k: v[:] for k, v in t_states.items()})
            state = self.rssm.img_step(states, action[:], training=False)
            feat = self.rssm.get_feat(state)
            print("horizon", h)
            for key, value in {**state, 'action': action, 'feat': feat}.items():
                # print()
                if key.startswith('t_'):
                    t_states[key] = value
                else:
                    seq[key].append(value)
                print(key, value.shape)
        seq = {k: tf.stack(v, 0) for k, v in seq.items()}
        if 'discount' in self.heads:
            disc = self.heads['discount'](seq['feat']).mean()
            if is_terminal is not None:
                # Override discount prediction for the first step with the true
                # discount factor from the replay buffer.
                true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
                true_first *= self.config.discount
                disc = tf.concat([true_first[:][None], disc[1:]], 0)
        else:
            disc = self.config.discount * tf.ones(seq['feat'].shape[:-1])
        seq['discount'] = disc
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.
        seq['weight'] = tf.math.cumprod(
            tf.concat([tf.ones_like(disc[:1]), disc[:-1]], 0), 0)
        return seq

    @tf.function(experimental_compile=False)
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

    @tf.function(experimental_compile=False)
    def video_pred(self, data, key, agent=None):
        print('data.keys()', data.keys())
        vb = self._video_pred_batches
        bf = min(self._bootstrap_frames, data['action'].shape[1] - 1)
        print("bootstrap_frames:", bf, data['action'].shape[1])
        decoder = self.heads['decoder']
        truth = data[key][:vb] + 0.5
        embed = self.encoder(data)
        transformer_embed = self.rssm.transformer_encode(data, tf.zeros_like(embed))
        states, _prior, _ = self.rssm.observe(
            embed[:vb, :bf], 
            transformer_embed[:vb, :bf] if transformer_embed is not None else None,
            data['image'][:vb, :bf],
            data['action'][:vb, :bf], 
            data['is_first'][:vb, :bf],
            training=False,
            transformer_weight=True
        )

        state_feat = self.rssm.get_feat(states)
        _prior_feat = self.rssm.get_feat(_prior)

        recon = decoder(state_feat)[key].mode()[:vb]
        recon_reward = self.heads['reward'](state_feat).mode()[:vb]
        recon_discount = self.heads['discount'](state_feat).mode()[:vb]
        if self._use_transformer_reward_head:
            recon_transformer_reward = self.heads['transformer_reward'](states['t_transformer']).mode()[:vb]
        if self._myopic_prediction:
            recon_myopic_reward = self.heads['myopic_reward'](states['myopic_out']).mode()[:vb]

        prior_recon = decoder(_prior_feat)[key].mode()[:vb]
        prior_recon_reward = self.heads['reward'](_prior_feat).mode()[:vb]
        prior_recon_discount = self.heads['discount'](_prior_feat).mode()[:vb]

        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data['action'][:vb, bf:], training=False, state=init, transformer_weight=True)

        prior_feat = self.rssm.get_feat(prior)

        openl = decoder(prior_feat)[key].mode()
        openl_reward = self.heads['reward'](prior_feat).mode()[:vb]
        openl_discount = self.heads['discount'](prior_feat).mode()[:vb]
        if self._use_transformer_reward_head:
            if 't_transformer' in prior:
                openl_transformer_reward = self.heads['transformer_reward'](prior['t_transformer']).mode()[:vb]
            else:
                openl_transformer_reward = tf.zeros_like(openl_reward)
            model_transformer_reward = tf.concat([recon_transformer_reward, openl_transformer_reward], 1)
        if self._myopic_prediction:
            openl_myopic_reward = self.heads['myopic_reward'](prior['myopic_out']).mode()[:vb]
            model_myopic_reward = tf.concat([recon_myopic_reward, openl_myopic_reward], 1)
        else:
            model_myopic_reward = None

        model_reward = tf.concat([recon_reward, openl_reward], 1)
        model_discount = tf.concat([recon_discount, openl_discount], 1)
        model = tf.concat([recon[:, :bf] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = tf.concat([truth, model, error], 2)
        prior_video = prior_recon[:, :bf] + 0.5  # [B, T, H, W, C]
        B, T, H, W, C = video.shape

        actions = data['action'][:vb]
        truth_reward = data['reward'][:vb]
        truth_discount = data['discount'][:vb]

        feat = tf.concat([state_feat, prior_feat], 1)

        ret_dict = {
            "images": video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C)),
            "prior_images": prior_video,
            "rewards": {
                "truth": truth_reward, 
                "model": model_reward,
                "prior": prior_recon_reward
            }, 
            "discounts": {
                "truth": truth_discount,
                "model": model_discount,
                "prior": prior_recon_discount
            },
            "actions": actions,
            "is_first": data['is_first'][:vb],
            "feat": feat,
        }

        if 'target' in data.keys():
            truth_dir = data['target'][:bf]
            try:
                model_dir = decoder(feat)['target'].mode()
                ret_dict['target'] = {
                    'truth': truth_dir,
                    'model': model_dir
                }
            except KeyError:
                pass
        
        if self._use_transformer_reward_head:
            ret_dict["rewards"]["transformer"] = model_transformer_reward

        if self._myopic_prediction:
            ret_dict["rewards"]["myopic"] = model_myopic_reward

        if self.config.rssm.use_transformer:
            ret_dict["transformer_weights"] = dict()
            for i in range(self.config.rssm.transformer.num_layers):
                weight_recon = states[f"t_weight_{i}"]
                try:
                    weight_openl = prior[f"t_weight_{i}"]
                    weight = tf.concat([weight_recon, weight_openl], 1)
                except KeyError:
                    weight = weight_recon
                ret_dict["transformer_weights"][i] = weight

            t_importance = self.calc_t_importance(ret_dict["transformer_weights"][self.config.rssm.transformer.num_layers - 1][:, :bf], 
                truth_reward[:, :bf], model_reward[:, :bf], model_transformer_reward[:, :bf], model_myopic_reward[:, :bf] if model_myopic_reward is not None else None)
            ret_dict["t_importance"] = t_importance

            if self.config.use_inside_transformer:
                memory_importance_recon = states["t_importance"]
                memory_importance_openl = prior["t_importance"]
                memory_importance = tf.concat([memory_importance_recon, memory_importance_openl], 1)
                ret_dict["memory_importance"] = memory_importance
        
        if self.config.use_int_reward:
            for k, head in self.heads.items():
                print("head", k, flush=True)
                if k.startswith('int_reward'):
                    int_reward = head(feat).mode()[:vb]
                    ret_dict[f'model_{k}'] = int_reward

        if agent is not None:
            recon_value = agent.critic(self.rssm.get_feat(states)).mode()[:vb]
            openl_value = agent.critic(self.rssm.get_feat(prior)).mode()[:vb]
            model_value = tf.concat([recon_value, openl_value], 1)
            ret_dict["value"] = model_value

        return ret_dict


class ActorCritic(common.Module):

    def __init__(self, config, step, num_actions):
        self.config = config
        self.step = step
        self.num_actions = num_actions
        self.actor = common.MLP(num_actions, **config.actor)
        self.critic = common.MLP([], **config.critic)
        if config.slow_target:
            self._target_critic = common.MLP([], **config.critic)
            self._updates = tf.Variable(0, tf.int64)
        else:
            self._target_critic = self.critic
        self.actor_opt = common.Optimizer('actor', **config.actor_opt)
        self.critic_opt = common.Optimizer('critic', **config.critic_opt)
        self.rewnorm = common.StreamNorm(**self.config.reward_norm)

    def train(self, world_model, start, is_terminal, reward_fn):
        print("in policy train()", flush=True)
        metrics = {}
        hor = self.config.imag_horizon
        # The weights are is_terminal flags for the imagination start states.
        # Technically, they should multiply the losses from the second trajectory
        # step onwards, which is the first imagined step. However, we are not
        # training the action that led into the first step anyway, so we can use
        # them to scale the whole sequence.
        with tf.GradientTape() as actor_tape:
            seq = world_model.imagine(self.actor, start, is_terminal, hor)
            reward = reward_fn(seq)
            seq['reward'], mets1 = self.rewnorm(reward)
            mets1 = {f'reward_{k}': v for k, v in mets1.items()}
            target, mets2 = self.target(seq)
            actor_loss, mets3 = self.actor_loss(seq, target)
        with tf.GradientTape() as critic_tape:
            critic_loss, mets4 = self.critic_loss(seq, target)
        metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
        metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
        metrics.update(**mets1, **mets2, **mets3, **mets4)
        self.update_slow_target()  # Variables exist after first forward pass.
        print("out policy train()", flush=True)
        return metrics

    def actor_loss(self, seq, target):
        # Actions:      0   [a1]  [a2]   a3
        #                  ^  |  ^  |  ^  |
        #                 /   v /   v /   v
        # States:     [z0]->[z1]-> z2 -> z3
        # Targets:     t0   [t1]  [t2]
        # Baselines:  [v0]  [v1]   v2    v3
        # Entropies:        [e1]  [e2]
        # Weights:    [ 1]  [w1]   w2    w3
        # Loss:              l1    l2
        metrics = {}
        # Two states are lost at the end of the trajectory, one for the boostrap
        # value prediction and one because the corresponding action does not lead
        # anywhere anymore. One target is lost at the start of the trajectory
        # because the initial state comes from the replay buffer.
        policy = self.actor(tf.stop_gradient(seq['feat'][:-2]))
        if self.config.actor_grad == 'dynamics':
            objective = target[1:]
        elif self.config.actor_grad == 'reinforce':
            baseline = self._target_critic(seq['feat'][:-2]).mode()
            advantage = tf.stop_gradient(target[1:] - baseline)
            objective = policy.log_prob(seq['action'][1:-1]) * advantage
        elif self.config.actor_grad == 'both':
            baseline = self._target_critic(seq['feat'][:-2]).mode()
            advantage = tf.stop_gradient(target[1:] - baseline)
            objective = policy.log_prob(seq['action'][1:-1]) * advantage
            mix = common.schedule(self.config.actor_grad_mix, self.step)
            objective = mix * target[1:] + (1 - mix) * objective
            metrics['actor_grad_mix'] = mix
        else:
            raise NotImplementedError(self.config.actor_grad)
        ent = policy.entropy()
        ent_scale = common.schedule(self.config.actor_ent, self.step)
        objective += ent_scale * ent
        weight = tf.stop_gradient(seq['weight'])
        actor_loss = -(weight[:-2] * objective).mean()
        metrics['actor_ent'] = ent.mean()
        metrics['actor_ent_scale'] = ent_scale
        return actor_loss, metrics

    def critic_loss(self, seq, target):
        # States:     [z0]  [z1]  [z2]   z3
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]   v3
        # Weights:    [ 1]  [w1]  [w2]   w3
        # Targets:    [t0]  [t1]  [t2]
        # Loss:        l0    l1    l2
        dist = self.critic(seq['feat'][:-1])
        target = tf.stop_gradient(target)
        weight = tf.stop_gradient(seq['weight'])
        critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
        metrics = {'critic': dist.mode().mean()}
        return critic_loss, metrics

    def target(self, seq):
        # States:     [z0]  [z1]  [z2]  [z3]
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]  [v3]
        # Discount:   [d0]  [d1]  [d2]   d3
        # Targets:     t0    t1    t2
        reward = tf.cast(seq['reward'], tf.float32)
        disc = tf.cast(seq['discount'], tf.float32)
        value = self._target_critic(seq['feat']).mode()
        # Skipping last time step because it is used for bootstrapping.
        target = common.lambda_return(
            reward[:-1], value[:-1], disc[:-1],
            bootstrap=value[-1],
            lambda_=self.config.discount_lambda,
            axis=0)
        metrics = {}
        metrics['critic_slow'] = value.mean()
        metrics['critic_target'] = target.mean()
        return target, metrics

    def update_slow_target(self):
        if self.config.slow_target:
            if self._updates % self.config.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                    self.config.slow_target_fraction)
                for s, d in zip(self.critic.variables, self._target_critic.variables):
                    d.assign(mix * s + (1 - mix) * d)
            self._updates.assign_add(1)
