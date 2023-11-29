###########################################################################################
# Training of Deep Q-Learning Networks (algo)
# Paper: https://www.nature.com/articles/nature14236
# Reference: https://github.com/Kchu/DeepRL_PyTorch
###########################################################################################
# The codes for the classical control game Pendulum are excerpted from OpenAI baseline Gym, 
# and an new image data output API is added to this game by defining a new observation wrapper 
# in wrappers.py.
# Parallel algo for an Inverted Pendulum with Image Data implemented in PyTorch and OpenAI Gym
import os
import torch
import pickle
import time
import argparse
import numpy as np 
from collections import deque

from parallel.parallelize import SubprocVecEnv
from envs.pendulum import PendulumEnv
from envs.wrappers import wrap_cover_pendulun_test


# 处理输入参数（游戏名称）
parser = argparse.ArgumentParser(description='Some settings of the experiment.')
parser.add_argument('--game', type=str, nargs=1, default='Pendulum', help='name of the games')
parser.add_argument('--algo', type=str, default='ddpg', help='the name of the algorithm')
parser.add_argument('--idling', help=' ', action='store_true', default=False)
args = parser.parse_args()
args.games = "".join(args.game)

###########   Environment Settings   ###########
# number of environments for C51
N_ENVS = 1
# Total simulation step
STEP_NUM = int(3e+5)
# visualize for agent playing
RENDERING = False
# openai gym env name
# ENV_NAME = args.game+'NoFrameskip-v4'
# 获取游戏名称
ENV_NAME = args.game
# idling
IDLING = args.idling

###########   Trainging Settings   ###########
# the number of channels for picture
CHANNEL_NUM = 4
# the dimension of velocity vector 
STATE_DIM = 1
# the dimension of action 
ACTION_DIM = 1 
# the maximum value of the action
MAX_ACTION = 2
# simulator steps for start learning
LEARN_START = int(1.5e+3)
# simulator steps for learning interval
LEARN_FREQ = 4
# epsilon-greedy
EPSILON = 1.0
# the length pf each episode
EPISODE_LENGTH = 250

###########   Save&Load Settings   ############
# check save/load
SAVE = True
LOAD = False
BUFFER_LOAD = False
# save frequency
SAVE_FREQ = int(1.5e+3) // N_ENVS

# paths for predction net, target net, result log
current_path = os.path.dirname(os.path.realpath("__file__"))
MODEL_PATH = os.path.join(current_path, 'data/model/'+args.algo+'_pred_net_o_'+args.games+'.pth')
RESULT_PATH = os.path.join(current_path, 'data/plots/'+args.algo+'_result_o_'+args.games+'.pkl')
BUFFER_PATH = os.path.join(current_path, 'data/replay_buffer/'+args.algo+'_buffer_'+args.games+'.pkl')

# create directory
if not os.path.exists(os.path.dirname(RESULT_PATH)):
    os.makedirs(os.path.dirname(RESULT_PATH))
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))
if not os.path.exists(os.path.dirname(BUFFER_PATH)):
    os.makedirs(os.path.dirname(BUFFER_PATH))

if args.algo == 'dqn':
    import algos.dqn
    from algos.dqn import DQN
    from misc.spaces import disc_actions 
    from envs.wrappers import wrap_cover_pendulun
    
    # create env
    single_env = wrap_cover_pendulun(ENV_NAME, disc_actions, episode_length=EPISODE_LENGTH)
    # define agent
    algo = DQN()
elif args.algo == 'ddpg':
    from algos.ddpg_image import DDPG
    from envs.wrappers import wrap_cover_pendulun_conti
    from envs.wrappers import wrap_cover_pendulun_super
    
    # create env
    single_env = wrap_cover_pendulun_conti(
        ENV_NAME, episode_length=EPISODE_LENGTH, fixed=True, repeat=6)
    # define agent 
    algo = DDPG(CHANNEL_NUM, STATE_DIM, ACTION_DIM, MAX_ACTION)
else:
    print("Undefined algorithm")
    raise

def anneal_epsilon():
    # annealing the epsilon(exploration strategy)
    if step <= int(1e+5 // N_ENVS):
        # linear annealing to 0.9 until million step
        EPSILON -= 0.9 / 1e+5 * N_ENVS
    elif step <= int(3e+5 // N_ENVS) and step > 1.5e+5:
    # else:
        # linear annealing to 0.99 until the end
        EPSILON -= 0.099 / 1.5e+5 * N_ENVS

def main():
    # create envs
    env = SubprocVecEnv([single_env for i in range(N_ENVS)])

    # model load with check
    if LOAD and os.path.isfile(MODEL_PATH):
        algo.load_model(MODEL_PATH)
        pkl_file = open(RESULT_PATH,'rb')
        result = pickle.load(pkl_file)
        pkl_file.close()
        print('Load complete!')
    else:
        result = []
        print('Initialize results!')

    print('Collecting experience...')

    # episode step for accumulate reward 
    epinfobuf = deque(maxlen=100)
    # check learning time
    start_time = time.time()

    # env reset
    s = env.reset()
    
    # Random choose action
    explore = True
    
    # for step in tqdm(range(1, STEP_NUM//N_ENVS+1)):
    for step in range(1, STEP_NUM // N_ENVS + 1):
        if step == 1e4:
            explore = False
        # When loading data from a file, don't need to sample from env.
        if args.algo == 'dqn':
            # annealing the epsilon(exploration strategy)
            anneal_epsilon()
            a = algo.choose_action(s, EPSILON)
        elif args.algo == 'ddpg':
            a = algo.choose_action(s, explore=explore)
        
        # take action and get next state
        s_, r, done, _, infos = env.step(a)
        if done.all(): 
            # log arrange
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: 
                    epinfobuf.append(maybeepinfo)
            s_ = env.reset()

        # store the transition
        algo.store_transition(s, a, r, s_, done, infos)
            
        s = s_
        if RENDERING:
            env.render()

        # if memory fill 50K and mod 4 = 0(for speed issue), learn pred net
        if (not IDLING) and (LEARN_START <= algo.memory_counter) and (algo.memory_counter % LEARN_FREQ == 0):
            loss = algo.learn()

        # print log and save
        if step % SAVE_FREQ == 0:
            # check time interval
            time_interval = round(time.time() - start_time, 2)
            # calc mean return
            mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]), 2)
            result.append(mean_100_ep_return)
            # print log
            if args.algo == 'dqn':
                print('Used Step: ', algo.memory_counter,
                    '| EPS: ', round(EPSILON, 3),
                    # '| Loss: ', loss,
                    '| Mean ep 100 return: ', mean_100_ep_return,
                    '| Used Time:',time_interval)
            elif args.algo == 'ddpg':
                print('Used Step: ', algo.memory_counter,
                    # '| Loss: ', loss,
                    '| Mean ep 100 return: ', mean_100_ep_return,
                    '| Used Time:',time_interval)
            else:
                print("Undefined algorithm")
                raise
            
            # if mean_100_ep_return >= -200:
            #     evaluate_performance(algo, disc_actions, episode_length=EPISODE_LENGTH)

            # save model
            # algo.save_model(algo, MODEL_PATH)
            pkl_file = open(RESULT_PATH, 'wb')
            pickle.dump(np.array(result), pkl_file)
            pkl_file.close()

    print("The training is done!")
    
def evaluate_performance(agent, render_mode="human", episode_length=200, test_num=1):
    # create env
    # 获取游戏名称
    game = args.game
    env = wrap_cover_pendulun_test(game, disc_actions, render_mode, episode_length)()
    
    for _ in range(test_num):
        # initialize the env
        s, _ = env.reset() 
        s = [s]
        # render
        env.render()
        
        for step in range(episode_length):
            a = agent.choose_action(s, 0.0, False)
            s_, r, done, infos, _ = env.step(a[0])
            s = [s_]
            env.render()
    
    
if __name__ == '__main__':
    main()