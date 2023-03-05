###########################################################################################
# Training of Deep Q-Learning Networks (DQN)
# Paper: https://www.nature.com/articles/nature14236
# Reference: https://github.com/Kchu/DeepRL_PyTorch
###########################################################################################
import os
import torch
import pickle
import time
import argparse
import numpy as np
from collections import deque

import dqn
from dqn import DQN
from wrappers import wrap, wrap_cover, wrap_cover_pendulun
from parallel import SubprocVecEnv


def main():
    
    # 处理输入参数（游戏名称）
    parser = argparse.ArgumentParser(description='Some settings of the experiment.')
    parser.add_argument('--game', type=str, nargs=1, 
                        help='name of the games. for example: Breakout'
                        )
    args = parser.parse_args()
    args.games = "".join(args.game)


    '''Environment Settings'''
    # number of environments for C51
    N_ENVS = 1
    # Total simulation step
    STEP_NUM = int(4e+5)
    # visualize for agent playing
    RENDERING = False
    # openai gym env name
    # ENV_NAME = args.game+'NoFrameskip-v4'
    ENV_NAME = args.game
    env = SubprocVecEnv([wrap_cover_pendulun(ENV_NAME) for i in range(N_ENVS)])

    '''Trainging Settings'''
    # simulator steps for start learning
    LEARN_START = int(1e+3)
    # simulator steps for learning interval
    LEARN_FREQ = 4
    # epsilon-greedy
    EPSILON = 1.0

    '''Save&Load Settings'''
    # check save/load
    SAVE = True
    LOAD = False
    # save frequency
    SAVE_FREQ = int(1e+3) // N_ENVS
    # paths for predction net, target net, result log
    current_path = os.path.dirname(os.path.realpath("__file__"))
    PRED_PATH = os.path.join(current_path, 'data/model/dqn_pred_net_o_'+args.games+'.pkl')
    TARGET_PATH = os.path.join(current_path, 'data/model/dqn_target_net_o_'+args.games+'.pkl')
    RESULT_PATH = os.path.join(current_path, 'data/plots/dqn_result_o_'+args.games+'.pkl')
    
    if not os.path.exists(os.path.dirname(RESULT_PATH)):
        os.makedirs(os.path.dirname(RESULT_PATH))
    
    if not os.path.exists(os.path.dirname(PRED_PATH)):
        os.makedirs(os.path.dirname(PRED_PATH))

    dqn = DQN()

    # model load with check
    if LOAD and os.path.isfile(PRED_PATH) and os.path.isfile(TARGET_PATH):
        dqn.load_model(PRED_PATH, TARGET_PATH)
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
    # print(s.shape)

    # for step in tqdm(range(1, STEP_NUM//N_ENVS+1)):
    for step in range(1, STEP_NUM // N_ENVS + 1):
        a = dqn.choose_action(s, EPSILON)
        # print('a',a)

        # take action and get next state
        s_, r, done, infos, _ = env.step(a)
        if done.all(): 
            # log arrange
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfobuf.append(maybeepinfo)
            s_ = env.reset()

        # store the transition
        for i in range(N_ENVS):
            dqn.store_transition(s[i], a[i], r[i], s_[i], done[i])

        # annealing the epsilon(exploration strategy)
        if step <= int(1e+5 / N_ENVS):
            # linear annealing to 0.9 until million step
            EPSILON -= 0.9 / 1e+5 * N_ENVS
        elif step <= int(2e+5 / N_ENVS):
        # else:
            # linear annealing to 0.99 until the end
            EPSILON -= 0.09 / 2e+5 * N_ENVS

        # if memory fill 50K and mod 4 = 0(for speed issue), learn pred net
        if (LEARN_START <= dqn.memory_counter) and (dqn.memory_counter % LEARN_FREQ == 0):
            loss = dqn.learn()

        # print log and save
        if step % SAVE_FREQ == 0:
            # check time interval
            time_interval = round(time.time() - start_time, 2)
            # calc mean return
            mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]), 2)
            result.append(mean_100_ep_return)
            # print log
            print('Used Step: ',dqn.memory_counter,
                '| EPS: ', round(EPSILON, 3),
                # '| Loss: ', loss,
                '| Mean ep 100 return: ', mean_100_ep_return,
                '| Used Time:',time_interval)
            # save model
            dqn.save_model(PRED_PATH, TARGET_PATH)
            pkl_file = open(RESULT_PATH, 'wb')
            pickle.dump(np.array(result), pkl_file)
            pkl_file.close()

        s = s_

        if RENDERING:
            env.render()
    print("The training is done!")
    
if __name__ == '__main__':
    main()
