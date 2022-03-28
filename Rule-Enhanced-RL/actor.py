import os
import pickle
import subprocess
import time
from argparse import ArgumentParser
from collections import deque
from itertools import count
from multiprocessing import Process, Array
from pathlib import Path
from typing import Tuple, Any
import copy
import random
import seaborn as sns
import pprint
import sys
import operator
import itertools

import numpy as np
import zmq
from pyarrow import serialize

from common import load_yaml_config, create_experiment_dir, Board, HiddenPrints, Snake
from utils import logger
from utils.cmdline import parse_cmdline_kwargs
from env.snake.env import SnakeEnv
gamma = 0.99
lam = 0.95

from custom_model import ACCNNModel
parser = ArgumentParser()
parser.add_argument('--alg', type=str, default='ppo', help='The RL algorithm')
parser.add_argument('--env', type=str, default='PongNoFrameskip-v4', help='The game environment')
parser.add_argument('--num_steps', type=int, default=10000000, help='The number of total training steps')
parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP address of learner server')
parser.add_argument('--data_port', type=int, default=5000, help='Learner server port to send training data')
parser.add_argument('--param_port', type=int, default=5001, help='Learner server port to subscribe model parameters')
parser.add_argument('--num_replicas', type=int, default=1, help='The number of actors')
parser.add_argument('--model', type=str, default='accnn', help='Training model')
parser.add_argument('--max_steps_per_update', type=int, default=128,
                    help='The maximum number of steps between each update')
parser.add_argument('--exp_path', type=str, default=None,
                    help='Directory to save logging data, model parameters and config file')
parser.add_argument('--num_saved_ckpt', type=int, default=10, help='Number of recent checkpoint files to be saved')
parser.add_argument('--max_episode_length', type=int, default=1000, help='Maximum length of trajectory')
parser.add_argument('--config', type=str, default=None, help='The YAML configuration file')
parser.add_argument('--use_gpu', action='store_true', help='Use GPU to sample every action')
parser.add_argument('--num_envs', type=int, default=10, help='The number of environment copies')


def run_one_agent(index, args, unknown_args, actor_status):
    from tensorflow.keras.backend import set_session
    import tensorflow.compat.v1 as tf

    # Set 'allow_growth'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # Connect to learner
    context = zmq.Context()
    context.linger = 0  # For removing linger behavior
    socket = context.socket(zmq.REQ)
    socket.connect(f'tcp://{args.ip}:{args.data_port}')

    # Initialize environment and agent instance
    # env, agent = init_components(args, unknown_args)

    # Configure logging only in one process
    if index == 0:
        logger.configure(str(args.log_path))
        # save_yaml_config(args.exp_path / 'config.yaml', args, 'actor', agent)
    else:
        logger.configure(str(args.log_path), format_strs=[])

    # Initialize values
    model_id = -1
    # episode_infos = deque(maxlen=100)
    num_episode = 0

    model = ACCNNModel(observation_space = [10, 20, 12], action_space = 4)
    env = SnakeEnv('snakes-3v3')


    while True:
        num_episode += 1
        # Do some updates
        # agent.update_sampling(update, nupdates)

        mb_states, mb_actions, mb_rewards, mb_dones, mb_values, mb_neglogp, mb_legalac = [], [], [], [], [], [], []
        start_time = time.time()
        done = False
        next_state, next_info = env.reset()
        # model.set_weights(init_weights)
        state_new = [0]*6
        while not done:
            # Sample action
            state = next_state
            info = next_info

            # 添加特征
            for i in range(6):
                # rl_ppo_act = rl_ppo(info['original_states'][i], [0, 1, 2, 3, 4, 5], False)
                board = get_board(info['original_states'][i], [0, 1, 2, 3, 4, 5], False)
                float_board = [ [0.0] * 20 for i in range(10)]
                # feature = 0.0
                # feature_max, feature_min = 0.0, 0.0

                # feature_max = board.max()
                # feature_min = board.min()
                # feature = max(feature_max, abs(feature_min))

                feature = 240.0
                for j in range(10):
                    for k in range(20):
                        float_board[j][k] = board[j][k] / feature
                # print("********start*********")
                # print("featuer_max is ", featuer_max)
                # print(feature)
                # print("float_board is", float_board)
                # print("********end*********")
                # board_normal = preprocessing.scale(board[1])
                board_normal = np.array(float_board)
                # print('board_normal is',board_normal)
                # file.writelines(str(board_normal))
                board_normal = board_normal.reshape(10, 20, 1)
                # file.writelines(str(board_normal))
                # file.close()
                state_new[i] = np.concatenate((state[i], board_normal), axis=2)
            state = np.array(state_new)

            action, v, p = model.forward(state, info["legal_action"])
            next_state, reward, done, next_info = env.step(action)
            if done:
                print("length:",info['length']," ave:",np.mean(info["length"]))
                logger.record_tabular("ave_rl", np.mean(info["length"]))

            mb_states.append(state)
            mb_actions.append(action)
            mb_rewards.append(reward)
            mb_dones.append(done)
            mb_values.append(v)
            mb_neglogp.append(p)
            mb_legalac.append(info["legal_action"])

            # for info_i in info:
            #     maybeepinfo = info_i.get('episode')
            #     if maybeepinfo:
            #         episode_infos.append(maybeepinfo)
            #         num_episode += 1

        mb_states = np.asarray(mb_states, dtype=state.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogp = np.asarray(mb_neglogp, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_legalac = np.asarray(mb_legalac, dtype=np.bool)

        mb_s, mb_a, mb_r, mb_d, mb_ex = [], [], [], [], []
        for j in range(6):
            for i in range(len(mb_states)):
                mb_s.append([mb_states[i][j]])
                mb_a.append([mb_actions[i][j]])
                mb_r.append([mb_rewards[i][j]])
                mb_d.append([mb_dones[i]])
                d = {}
                d['value'] = [mb_values[i][j]]
                d['neglogp'] = [mb_neglogp[i][j]]
                d['legal_action'] = [mb_legalac[i][j]]
                mb_ex.append(d)

        mb_s = np.asarray(mb_s, dtype=state.dtype)
        mb_r = np.asarray(mb_r, dtype=np.float32)
        mb_a = np.asarray(mb_a)
        mb_d = np.asarray(mb_d, dtype=np.bool)

        data = prepare_training_data([mb_s, mb_a, mb_r, mb_d, state, mb_ex])
        # data = prepare_training_data([mb_states, mb_actions, mb_rewards, mb_dones, state, mb_extras])
        # data = agent.prepare_training_data([mb_states, mb_actions, mb_rewards, mb_dones, state, mb_extras])
        # post_processed_data = agent.post_process_training_data(data)
        socket.send(serialize(data).to_buffer())
        socket.recv()

        # send_data_interval = time.time() - start_time
        # Log information
        logger.record_tabular("episodes", num_episode)
        logger.dump_tabular()

        # Update weights
        new_weights, model_id = find_new_weights(model_id, args.ckpt_path)
        if new_weights is not None:
            model.set_weights(new_weights)

    # actor_status[index] = 1


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def prepare_training_data(trajectory):
    mb_states, mb_actions, mb_rewards, mb_dones, next_state, mb_extras = trajectory
    mb_values = np.asarray([extra_data['value'] for extra_data in mb_extras])
    mb_neglogp = np.asarray([extra_data['neglogp'] for extra_data in mb_extras])
    mb_legalac = np.asarray([extra_data['legal_action'] for extra_data in mb_extras])
    last_values = [[0]]
    mb_values = np.concatenate([mb_values, last_values])

    mb_deltas = mb_rewards + gamma * mb_values[1:] * (1.0 - mb_dones) - mb_values[:-1]

    nsteps = len(mb_states)
    mb_advs = np.zeros_like(mb_rewards)
    lastgaelam = 0
    for t in reversed(range(nsteps)):
        nextnonterminal = 1.0 - mb_dones[t]
        mb_advs[t] = lastgaelam = mb_deltas[t] + gamma * lam * nextnonterminal * lastgaelam

    mb_returns = mb_advs + mb_values[:-1]
    data = [sf01(arr) for arr in [mb_states, mb_returns, mb_actions, mb_values[:-1], mb_neglogp, mb_legalac]]
    name = ['state', 'return', 'action', 'value', 'neglogp', 'legal_action']
    return dict(zip(name, data))


def run_weights_subscriber(args, actor_status):
    """Subscribe weights from Learner and save them locally"""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f'tcp://{args.ip}:{args.param_port}')
    socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe everything

    for model_id in count(1):  # Starts from 1
        while True:
            try:
                weights = socket.recv(flags=zmq.NOBLOCK)

                # Weights received
                with open(args.ckpt_path / f'{model_id}.{args.alg}.{args.env}.ckpt', 'wb') as f:
                    f.write(weights)

                if model_id > args.num_saved_ckpt:
                    os.remove(args.ckpt_path / f'{model_id - args.num_saved_ckpt}.{args.alg}.{args.env}.ckpt')
                break
            except zmq.Again:
                pass

            if all(actor_status):
                # All actors finished works
                return

            # For not cpu-intensive
            time.sleep(1)


def find_new_weights(current_model_id: int, ckpt_path: Path) -> Tuple[Any, int]:
    try:
        ckpt_files = sorted(os.listdir(ckpt_path), key=lambda p: int(p.split('.')[0]))
        latest_file = ckpt_files[-1]
    except IndexError:
        # No checkpoint file
        return None, -1
    new_model_id = int(latest_file.split('.')[0])

    if int(new_model_id) > current_model_id:
        loaded = False
        while not loaded:
            try:
                with open(ckpt_path / latest_file, 'rb') as f:
                    new_weights = pickle.load(f)
                    if int(new_model_id) % 100 == 0:
                        with open( '/home/wangjt/snakes/rl-framework-baseline-feature/feature_model_backup_1_2000_data1223_teamreward_2/' f'{int(new_model_id)}.pth',
                                    'wb') as f:
                                pickle.dump(new_weights, f)
                loaded = True
            except (EOFError, pickle.UnpicklingError):
                # The file of weights does not finish writing
                pass

        return new_weights, new_model_id
    else:
        return None, current_model_id

def get_board(observation_list, action_space_list, is_act_continuous):
    observation_len = len(observation_list.keys())
    teams = None
    teams = [[0, 1, 2], [3, 4, 5]]  # 3v3
    teams_count = len(teams)
    snakes_count = sum([len(_) for _ in teams])

    # read observation
    obs = observation_list.copy()
    board_height = obs['board_height']  # 10
    board_width = obs['board_width']  # 20
    # print("obs['controlled_snake_index'] is ", obs['controlled_snake_index'])
    ctrl_agent_index = obs['controlled_snake_index'] - 2  # 0, 1, 2, 3, 4, 5
    # last_directions = obs['last_direction']  # ['up', 'left', 'down', 'left', 'left', 'left']
    beans_positions = obs[1]  # e.g.[[7, 15], [4, 14], [5, 12], [4, 12], [5, 7]]
    snakes = {key - 2: Snake(obs[key], board_height, board_width, beans_positions)
              for key in obs.keys() & {_ + 2 for _ in range(snakes_count)}}  # &: intersection
    team_indexes = [_ for _ in teams if ctrl_agent_index in _][0]

    init_board = Board(board_height, board_width, snakes, beans_positions, teams)
    bd = copy.deepcopy(init_board)

    with HiddenPrints():
        while not all(_ == [] for _ in bd.open.values()):  # loop until all values in open are empty list
            bd.step()
    # print(bd.board)
    return bd.board


def main():
    # Parse input parameters
    args, unknown_args = parser.parse_known_args()
    args.num_steps = int(args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)

    # Load config file
    load_yaml_config(args, 'actor')

    # Create experiment directory
    create_experiment_dir(args, 'ACTOR-')

    args.ckpt_path = args.exp_path / 'ckpt'
    args.log_path = args.exp_path / 'log'
    args.ckpt_path.mkdir()
    args.log_path.mkdir()

    # Record commit hash
    with open(args.exp_path / 'hash', 'w') as f:
        f.write(str(subprocess.run('git rev-parse HEAD'.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')))

    # Disable GPU
    if not args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Running status of actors
    actor_status = Array('i', [0] * args.num_replicas)

    # Run weights subscriber
    subscriber = Process(target=run_weights_subscriber, args=(args, actor_status))
    subscriber.start()

    def exit_wrapper(index, *x, **kw):
        """Exit all agents on KeyboardInterrupt (Ctrl-C)"""
        try:
            run_one_agent(index, *x, **kw)
        except KeyboardInterrupt:
            if index == 0:
                for _i, _p in enumerate(agents):
                    if _i != index:
                        _p.terminate()
                    actor_status[_i] = 1

    agents = []
    for i in range(args.num_replicas):
        p = Process(target=exit_wrapper, args=(i, args, unknown_args, actor_status))
        p.start()
        os.system(f'taskset -p -c {(i+20) % os.cpu_count()} {p.pid}')  # For CPU affinity

        agents.append(p)

    for agent in agents:
        agent.join()

    subscriber.join()


if __name__ == '__main__':
    main()
