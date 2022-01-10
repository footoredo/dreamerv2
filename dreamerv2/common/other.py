import collections
import contextlib
import re
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from . import dists
from . import tfutils


class RandomAgent:

    def __init__(self, num_actions, discrete, logprob=False):
        self._logprob = logprob
        if discrete:
            self._dist = dists.OneHotDist(tf.zeros(num_actions))
        else:
            dist = tfd.Uniform(-np.ones(num_actions), np.ones(num_actions))
            self._dist = tfd.Independent(dist, 1)

    def __call__(self, obs, state=None, mode=None):
        action = self._dist.sample(len(obs['is_first']))
        output = {'action': action}
        if self._logprob:
            output['logprob'] = self._dist.log_prob(action)
        return output, None


def static_scan(fn, inputs, start, reverse=False):
    last = start
    outputs = [[] for _ in tf.nest.flatten(start)]
    indices = range(tf.nest.flatten(inputs)[0].shape[0])
    if reverse:
        indices = reversed(indices)
    for index in indices:
        inp = tf.nest.map_structure(lambda x: x[index], inputs)
        last = fn(last, inp)
        [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
        # print(index, len(outputs))
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    # def nested_stack(x):
    #     if type(x[0]) == list:
    #         x = [nested_stack(_x) for _x in x]
    #     return tf.stack(x, 0)
    outputs = [tf.stack(x, 0) for x in outputs]
    # for i, x in enumerate(outputs):
    #     print(type(x))
    return tf.nest.pack_sequence_as(start, outputs)


def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        step = tf.cast(step, tf.float32)
        match = re.match(r'linear\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = tf.clip_by_value(step / duration, 0, 1)
            return (1 - mix) * initial + mix * final
        match = re.match(r'warmup\((.+),(.+)\)', string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = tf.clip_by_value(step / warmup, 0, 1)
            return scale * value
        match = re.match(r'exp\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        match = re.match(r'horizon\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = tf.clip_by_value(step / duration, 0, 1)
            horizon = (1 - mix) * initial + mix * final
            return 1 - 1 / horizon
        raise NotImplementedError(string)


def lambda_return(
        reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * tf.ones_like(reward)
    dims = list(range(reward.shape.ndims))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        reward = tf.transpose(reward, dims)
        value = tf.transpose(value, dims)
        pcont = tf.transpose(pcont, dims)
    if bootstrap is None:
        bootstrap = tf.zeros_like(value[-1])
    next_values = tf.concat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns = static_scan(
        lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
        (inputs, pcont), bootstrap, reverse=True)
    if axis != 0:
        returns = tf.transpose(returns, dims)
    return returns


def action_noise(action, amount, discrete):
    if amount == 0:
        return action
    amount = tf.cast(amount, action.dtype)
    if discrete:
        probs = amount / action.shape[-1] + (1 - amount) * action
        return dists.OneHotDist(probs=probs).sample()
    else:
        return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)


class StreamNorm(tfutils.Module):

    def __init__(self, shape=(), momentum=0.99, scale=1.0, eps=1e-8):
        # Momentum of 0 normalizes only based on the current batch.
        # Momentum of 1 disables normalization.
        self._shape = tuple(shape)
        self._momentum = momentum
        self._scale = scale
        self._eps = eps
        self.mag = tf.Variable(tf.ones(shape, tf.float64), False)

    def __call__(self, inputs):
        metrics = {}
        self.update(inputs)
        metrics['mean'] = inputs.mean()
        metrics['std'] = inputs.std()
        outputs = self.transform(inputs)
        metrics['normed_mean'] = outputs.mean()
        metrics['normed_std'] = outputs.std()
        return outputs, metrics

    def reset(self):
        self.mag.assign(tf.ones_like(self.mag))

    def update(self, inputs):
        batch = inputs.reshape((-1,) + self._shape)
        mag = tf.abs(batch).mean(0).astype(tf.float64)
        self.mag.assign(self._momentum * self.mag + (1 - self._momentum) * mag)

    def transform(self, inputs):
        values = inputs.reshape((-1,) + self._shape)
        values /= self.mag.astype(inputs.dtype)[None] + self._eps
        values *= self._scale
        return values.reshape(inputs.shape)


class Timer:

    def __init__(self):
        self._indurs = collections.defaultdict(list)
        self._outdurs = collections.defaultdict(list)
        self._start_times = {}
        self._end_times = {}

    @contextlib.contextmanager
    def section(self, name):
        self.start(name)
        yield
        self.end(name)

    def wrap(self, function, name):
        def wrapped(*args, **kwargs):
            with self.section(name):
                return function(*args, **kwargs)

        return wrapped

    def start(self, name):
        now = time.time()
        self._start_times[name] = now
        if name in self._end_times:
            last = self._end_times[name]
            self._outdurs[name].append(now - last)

    def end(self, name):
        now = time.time()
        self._end_times[name] = now
        self._indurs[name].append(now - self._start_times[name])

    def result(self):
        metrics = {}
        for key in self._indurs:
            indurs = self._indurs[key]
            outdurs = self._outdurs[key]
            metrics[f'timer_count_{key}'] = len(indurs)
            metrics[f'timer_inside_{key}'] = np.sum(indurs)
            metrics[f'timer_outside_{key}'] = np.sum(outdurs)
            indurs.clear()
            outdurs.clear()
        return metrics


class CarryOverState:

    def __init__(self, fn):
        self._fn = fn
        self._state = None

    def __call__(self, *args):
        self._state, out = self._fn(*args, self._state)
        # print(self._state)
        return out



class RunningStats:
    
    def __init__(self, shape, decay=0.99):
        self._shape = shape
        self._decay = decay
        self.reset()
        
    def reset(self):
        self._moving_mean = tf.Variable(tf.zeros(self._shape))
        self._moving_variance = tf.Variable(tf.zeros(self._shape))
        self._zero_debias_count = tf.Variable(0)
        self._min = tf.Variable(tf.ones(self._shape) * 1e9)
        self._first = True
    
    def push(self, x):
        axis = tuple(range(len(x.shape) - len(self._shape)))
        tfp.stats.assign_moving_mean_variance(
            value=tf.stop_gradient(x), 
            moving_mean=self._moving_mean, 
            moving_variance=self._moving_variance,
            zero_debias_count=self._zero_debias_count,
            decay=self._decay,
            axis=axis
        )
        self._min = tf.minimum(self._min, tf.reduce_min(x, axis=axis))
        self._first = False

    def mean(self):
        return self._moving_mean
    
    def min(self):
        return self._min

    def var(self):
        return self._moving_variance
    
    def std(self):
        return tf.sqrt(self.var())
    
    def norm(self, x, bias=1e-8):
        return (x - self.mean()) / (self.std() + bias)
    
    def half_norm(self, x, bias=1e-8):  # normalize to half normal distribution:
        return (x - self.min()) / (self.std() + bias)
        