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
        if state is None:
            latent = self.wm.rssm.initial(len(obs['reward']))
            action = tf.zeros((len(obs['reward']), self._num_act))
            state = latent, action
        latent, action = state
        embed = self.wm.encoder(self.wm.preprocess(obs))
        sample = (mode == 'train') or not self.config.eval_state_mean
        latent, _ = self.wm.rssm.obs_step(
            latent, action, embed, obs['is_first'], sample)
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
        action = common.action_noise(action, noise, self.config.discrete)

        # mat = np.eye(17)
        # mat[7, 7] = 0
        # mat[7, 0] = 1  # replace place_stone to 
        # mat = tf.constant(mat, dtype=action.dtype)
        # action = tf.matmul(action, mat)
        
        outputs = {'action': action}
        state = (latent, action)
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
            r = self.wm.heads['reward'](seq['feat']).mode()
            if self.config.use_int_reward:
                r += self.config.int_reward_coef * self.wm.heads['int_reward'](seq['feat']).mode()
            return r
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
        rtn = None
        for key in data:
            if re.match(self.config.decoder.cnn_keys, key):
                name = key.replace('/', '_')
                pred_dict = self.wm.video_pred(data, key, self._task_behavior)
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
        self.rssm = common.EnsembleRSSM(num_actions=num_actions, **config.rssm)
        self.encoder = common.Encoder(**config.encoder)
        self.heads = {}
        self.heads['decoder'] = common.Decoder(shapes, **config.decoder)
        self.heads['reward'] = common.MLP([], **config.reward_head)
        self._use_int_reward = config.use_int_reward
        if self._use_int_reward:
            print("use int reward!", flush=True)
            self.heads['int_reward'] = common.MLP([], **config.reward_head)
        # print("wm.reward:", self.heads['reward'].variables)
        if config.pred_discount:
            self.heads['discount'] = common.MLP([], **config.discount_head)
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

    def loss(self, data, state=None):
        # print("in loss()", flush=True)
        data = self.preprocess(data)
        # print("1", flush=True)
        embed = self.encoder(data)
        # print("2", flush=True)
        post, prior = self.rssm.observe(
            embed, data['action'], data['is_first'], training=True, state=state)
        # print("3", flush=True)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        assert len(kl_loss.shape) == 0
        likes = {}
        losses = {'kl': kl_loss}
        feat = self.rssm.get_feat(post)
        # print(feat.shape)
        for name, head in self.heads.items():
            if name == "int_reward": 
                continue
            grad_head = (name in self.config.grad_heads)
            inp = feat if grad_head else tf.stop_gradient(feat)
            out = head(inp)
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                print(key, data[key].shape, dist.mean().shape)
                like = tf.cast(dist.log_prob(data[key]), tf.float32)
                # if not key in self._running_stats:
                #     self._running_stats[key] = common.RunningStats(like.shape)
                # self._running_stats[key].push(-like)
                likes[key] = like
                losses[key] = -like.mean()
        
        metrics = {}
        if self._use_int_reward:
            model_like = 0
            print("in calc int")
            for k, v in likes.items():
                _v = (v.mean() - v) / (v.std() + 1e-8)
                mask = tf.cast(_v > 1, tf.float32)
                _v = mask * _v  # only keep significant reward (> 1 std)
                print(k, _v.shape)
                metrics[f'{k}_like_max'] = _v.max()
                metrics[f'{k}_like_min'] = _v.min()
                metrics[f'{k}_like_std'] = _v.std()
                model_like += self.config.int_reward_scales.get(k, 0.0) * _v
            int_reward = model_like
            data["int_reward"] = int_reward
            inp = tf.stop_gradient(feat)
            # inp = feat
            dist = self.heads["int_reward"](inp)
            like = tf.cast(dist.log_prob(int_reward), tf.float32)
            # print("model_like", model_like.shape, inp.shape, like.shape, flush=True)
            likes["int_reward"] = like
            losses["int_reward"] = -like.mean()
            metrics['int_reward_max'] = int_reward.max()
            metrics['int_reward_min'] = int_reward.min()
            metrics['int_reward_mean'] = int_reward.mean()
            metrics['int_reward_std'] = int_reward.std()
            
        model_loss = 0
        if self._use_int_reward:
            model_loss += losses["int_reward"]
        # print("losses:", flush=True)
        # model_loss = tf.zeros([], dtype=tf.float32)
        for k, v in losses.items():
            # print(k, v.shape, self.config.loss_scales.get(k, 1.0), flush=True)
            model_loss += self.config.loss_scales.get(k, 1.0) * v
            
        # model_loss = sum(
        #     self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
        outs = dict(
            embed=embed, feat=feat, post=post,
            prior=prior, likes=likes, kl=kl_value)
        metrics.update({f'{name}_loss': value for name, value in losses.items()})
        metrics['model_kl'] = kl_value.mean()
        metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
        metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
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
        vb = self._video_pred_batches
        bf = self._bootstrap_frames
        decoder = self.heads['decoder']
        truth = data[key][:vb] + 0.5
        embed = self.encoder(data)
        states, _ = self.rssm.observe(
            embed[:vb, :bf], 
            data['action'][:vb, :bf], 
            data['is_first'][:vb, :bf],
            training=False
        )

        state_feat = self.rssm.get_feat(states)

        recon = decoder(state_feat)[key].mode()[:vb]
        recon_reward = self.heads['reward'](state_feat).mode()[:vb]
        recon_discount = self.heads['discount'](state_feat).mode()[:vb]

        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data['action'][:vb, bf:], training=False, state=init)

        prior_feat = self.rssm.get_feat(prior)

        openl = decoder(prior_feat)[key].mode()
        openl_reward = self.heads['reward'](prior_feat).mode()[:vb]
        openl_discount = self.heads['discount'](prior_feat).mode()[:vb]

        model_reward = tf.concat([recon_reward, openl_reward], 1)
        model_discount = tf.concat([recon_discount, openl_discount], 1)
        model = tf.concat([recon[:, :bf] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = tf.concat([truth, model, error], 2)
        B, T, H, W, C = video.shape

        actions = data['action'][:vb]
        truth_reward = data['reward'][:vb]
        truth_discount = data['discount'][:vb]

        feat = tf.concat([state_feat, prior_feat], 1)

        ret_dict = {
            "images": video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C)),
            "rewards": {
                "truth": truth_reward, 
                "model": model_reward
            }, 
            "discounts": {
                "truth": truth_discount,
                "model": model_discount
            },
            "actions": actions,
            "is_first": data['is_first'][:vb],
            "feat": feat,
        }
        
        if self.config.use_int_reward:
            int_reward = self.heads['int_reward'](feat).mode()[:vb]
            ret_dict['model_int_reward'] = int_reward

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
