import collections
import logging
import os
import pathlib
import re
import sys
import warnings
import shutil
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ["TF_GPU_THREAD_COUNT"] = "4"
logging.getLogger().setLevel('DEBUG')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))

import numpy as np
import ruamel.yaml as yaml
import joblib
import tensorflow as tf

import agent
import reward_trainer
import common

from common import Config
from common import GymWrapper
from common import Flags
from common import ResizeImage
from common import NormalizeAction
from common import OneHotAction
from common import TimeLimit
from common import TerminalOutput
from common import JSONLOutput
from common import TensorBoardOutput

configs = yaml.safe_load(
    (pathlib.Path(__file__).parent / 'configs.yaml').read_text())
defaults = common.Config(configs.pop('defaults'))
configs = common.Config({k: defaults.update(v) for k, v in configs.items()})


def train(env, config, outputs=None):
    tf.random.set_seed(config.seed)
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / 'config.yaml')
    print(config, '\n')
    print('Logdir', logdir)

    tf.config.experimental_run_functions_eagerly(not config.jit)
    message = 'No GPU found. To actually train on CPU remove this assert.'
    assert tf.config.experimental.list_physical_devices('GPU'), message
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        from tensorflow.keras.mixed_precision import experimental as prec
        prec.set_policy(prec.Policy('mixed_float16'))

    outputs = outputs or [
        common.TerminalOutput(),
        common.JSONLOutput(config.logdir),
        common.TensorBoardOutput(config.logdir),
    ]
    if config.use_wandb:
        outputs.append(common.WanDBOutput(config.wandb_config, config))
    replay = common.Replay(logdir / 'train_episodes', seed=config.seed, **config.replay)
    step = common.Counter(replay.stats['total_steps'])
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_train = common.Every(config.train_every)
    should_log = common.Every(config.log_every)
    should_video = common.Every(config.log_every)
    should_save = common.Every(config.save_every)
    should_expl = common.Until(config.expl_until)

    def per_episode(ep):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(f'Episode has {length} steps and return {score:.1f}.')
        logger.scalar('return', score)
        logger.scalar('length', length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f'sum_{key}', ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f'mean_{key}', ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f'max_{key}', ep[key].max(0).mean())
        if should_video(step):
            for key in config.log_keys_video:
                logger.video(f'policy_{key}', ep[key])
        logger.add(replay.stats)
        logger.write()

    print('Create envs.')
    act_space = env.act_space['action']
    num_actions = act_space.n if config.discrete else act_space.shape[-1]
    driver = common.Driver([env])
    driver.on_episode(per_episode)
    driver.on_step(lambda tran, worker: step.increment())
    driver.on_step(replay.add_step)
    driver.on_reset(replay.add_step)

    prefill = max(0, config.prefill - replay.stats['total_steps'])
    if prefill:
        print(f'Prefill dataset ({prefill} steps).')
        random_agent = common.RandomAgent(num_actions, config.discrete)
        driver(random_agent, steps=prefill, episodes=1)
        driver.reset()

    print('Create agent.', flush=True)
    dataset = iter(replay.dataset(**config.dataset))
    shapes = {k: v.shape[2:] for k, v in dataset.element_spec.items()}
    agnt = agent.Agent(config, logger, step, shapes)
    train_agent = common.CarryOverState(agnt.train)
    # tf.profiler.experimental.start(str(logdir))
    if config.make_graph:
        model_loss = agnt.train.get_concrete_function(next(dataset))
        graph = model_loss.graph
        writer = tf.summary.create_file_writer(str(logdir))
        with writer.as_default():
            tf.summary.graph(graph)
            tf.summary.flush()
        return
    else:
        train_agent(next(dataset))
    # model_loss, prior = agnt.train(next(dataset))
    # print(model_loss)
    # print(weights)
    # print(mask, flush=True)
    # print(t_hidden, flush=True)
    # _prior = dict()
    # for k, v in prior.items():
    #     _prior[k] = v.numpy()
    # joblib.dump(_prior, logdir / "prior.data")
    # return
    # model_loss = agnt.train.get_concrete_function(next(dataset))
    # graph = model_loss.graph
    # # logger.write()
    # writer = tf.summary.create_file_writer(str(logdir))
    # with writer.as_default():
    #     tf.summary.graph(graph)
    #     tf.summary.flush()
    # tf.summary.flush()
    # return
    # for node in graph.as_graph_def().node:
    #     print(f'{node.input} -> {node.name}')
    # logger.write()
    # logger.write()
    # with writer.as_default():
    #     tf.summary.trace_export(
    #     name="my_func_trace",
    #     profiler_outdir=str(logdir))
    # logger.write()
    # tf.profiler.experimental.stop()
    # return
    if (logdir / 'variables.pkl').exists():
        agnt.load(logdir / 'variables.pkl')
    else:
        print('Pretrain agent.')
        for _ in range(config.pretrain):
            train_agent(next(dataset))
    policy = lambda *args: agnt.policy(
        *args, mode='explore' if should_expl(step) else 'train')
    print("before train", flush=True)

    def train_step(tran, worker):
        if should_train(step):
            for i in range(config.train_steps):
                print("train", step.value, i, flush=True)
                mets = train_agent(next(dataset))
                print("finished", step.value, i, flush=True)
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(agnt.report(next(dataset)))
            logger.write(fps=True)
        if should_save(step):
            agnt.save(logdir / f'variables-{step.value}.pkl')

    driver.on_step(train_step)

    if config.benchmark:
        tf.profiler.experimental.start(str(logdir))
    while step < config.steps:
        print("step", step.value)
        logger.write()
        driver(policy, steps=config.eval_every)
        agnt.save(logdir / 'tmp_variables.pkl')
        if config.save_episodes:
            saved_files = replay.save_episodes(logdir / "tmp_train_episodes")
            for filename in saved_files:
                os.rename(logdir / "tmp_train_episodes" / filename, logdir / "train_episodes" / filename)
        os.rename(logdir / 'tmp_variables.pkl', logdir / 'variables.pkl')

        # agnt.save(logdir / f'variables-{step.value}.pkl')
        # os.symlink(str(logdir / f'variables-{step.value}.pkl'), str(logdir / 'variables.pkl.tmp'))
        # os.rename(str(logdir / 'variables.pkl.tmp'), str(logdir / 'variables.pkl'))  # to avoid FileExisError
    if config.benchmark:
        tf.profiler.experimental.stop()
        
    print("Done!")


def replay(env, config, outputs=None, actor_mode='train'):
    # os.environ['TF_DETERMINISTIC_OPS'] = 'true'
    # tf.config.set_visible_devices([], 'GPU')
    tf.random.set_seed(config.seed)
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / 'config.yaml')
    print(config, '\n')
    print('Logdir', logdir)
    print(f"Replay... seed={config.seed}")

    tf.config.experimental_run_functions_eagerly(True)
    message = 'No GPU found. To actually train on CPU remove this assert.'
    assert tf.config.experimental.list_physical_devices('GPU'), message
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        from tensorflow.keras.mixed_precision import experimental as prec
        prec.set_policy(prec.Policy('mixed_float16'))

    writer = tf.summary.create_file_writer(str(logdir))

    tf.debugging.set_log_device_placement(True)
    
    # tf.config.run_functions_eagerly(True)

    # prefill_steps = 10

    outputs = outputs or [
        common.TerminalOutput()
    ]
    prefill_replay = common.Replay(logdir / 'prefill_episodes', seed=config.seed, **config.replay)
    prefill_step = common.Counter(prefill_replay.stats['total_steps'])

    print('Create envs.')
    act_space = env.act_space['action']
    num_actions = act_space.n if config.discrete else act_space.shape[-1]
    prefill_driver = common.Driver([env])
    prefill_driver.on_step(lambda tran, worker: prefill_step.increment())
    prefill_driver.on_step(prefill_replay.add_step)
    prefill_driver.on_reset(prefill_replay.add_step)

    random_agent = common.RandomAgent(num_actions, config.discrete)
    prefill_driver(random_agent, steps=1, episodes=1)

    print("Prefill ended.", flush=True)

    replay = common.Replay(logdir / 'train_episodes', seed=config.seed, start_with_first=True, indexed_sampling=True, **config.replay)
    step = common.Counter(replay.stats['total_steps'])
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)

    print('Create agent.', flush=True)
    prefill_dataset = iter(prefill_replay.dataset(1, 5))
    # print('prefill_dataset')
    # __data = next(prefill_dataset)
    # for key, item in __data.items():
    #     # print(key, item)
    #     print(item.device)
    #     break
    shapes = {k: v.shape[2:] for k, v in prefill_dataset.element_spec.items()}
    agnt = agent.Agent(config, logger, step, shapes)
    print('before init train', flush=True)
    policy = lambda *args: agnt.policy(
        *args, mode='train')
    prefill_driver(policy, steps=1, episodes=1)
    agnt.train(next(prefill_dataset))
    prefill_dataset = iter(prefill_replay.dataset(1, 5))
    # agnt.policy()
    # _, _ = agnt.report(next(prefill_dataset), True)
    # return
    print('before load', flush=True)
    agnt.load(logdir / 'variables.pkl')

    print('Load variables ended', flush=True)

    policy = lambda *args: agnt.policy(
        *args, mode=actor_mode)

    try:
        env.reset_episode()
    except ValueError:
        pass
    # _env = env
    # while True:
    #     try:
    #         _env.reset_episode()
    #         continue
    #     except:
    #         _env = _env._env

    driver = common.Driver([env])
    driver.on_step(lambda tran, worker: step.increment())
    driver.on_step(replay.add_step)
    driver.on_reset(replay.add_step)

    # print("step:", step.value, flush=True)
    # while step < config.steps:
    #     print("step:", step.value, flush=True)
    driver(policy, steps=config.steps, episodes=config.dataset.batch)
        # dataset = iter(replay.dataset(**config.dataset))

    print("data collected!", flush=True)

    dataset = iter(replay.dataset(**config.dataset))
    tf.profiler.experimental.start(str(logdir))
    _, data = agnt.report(next(dataset), True)
    tf.profiler.experimental.stop()
    save_path = logdir / 'replay.data'

    def to_numpy(tensor_dict):
        numpy_dict = dict()
        for name, value in tensor_dict.items():
            if type(value) == dict:
                numpy_dict[name] = to_numpy(value)
            else:
                numpy_dict[name] = value.numpy()
        return numpy_dict

    import joblib
    joblib.dump(to_numpy(data), save_path)

    print("Done!")


def train_reward(config, replay_dir):
    replay_dir = pathlib.Path(replay_dir).expanduser()
    replay = common.Replay(replay_dir, seed=config.seed, start_with_first=True, indexed_sampling=True, **config.replay)
    trainer = reward_trainer.RewardTrainer(config)
    dataset = iter(replay.dataset(**config.reward_pred_dataset))

    for i in range(1000):
        loss = trainer.train(next(dataset))
        print(f"Loss at step {i}: {loss}", flush=True)


    infos = []
    for i in range(20):
        data, pred, weight = trainer.test(next(dataset))
        infos.append((data, pred, weight))

    joblib.dump(infos, "infos.data")

    print("Done!")

    # cnt = 0
    # while cnt < 20:
    #     data, pred, weight = trainer.test(next(dataset))
    #     for i in range(pred.shape[0]):
    #         for j in range(pred.shape[1]):
    #             if data['reward'][i, j] > 0.4:
    #                 cnt += 1
    #                 print(data['reward'][i, j], pred[i, j])
    #                 print(weight['dec0'][i, :, j])
    #                 print(weight['dec1'][i, :, j])