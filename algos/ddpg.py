import sys
import os
import argparse
from itertools import count
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

current_path = os.path.dirname(os.path.realpath("__file__"))
abs_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(abs_path)

from buffers.replay_memory import ReplayBufferImage
from misc.ou_noise import OUNoise
from misc.generate_obs import get_reward, get_next_obs


'''DDPG settings'''
# target policy sync interval
TARGET_REPLACE_ITER = 2
# (prioritized) experience replay memory size
MEMORY_CAPACITY = int(1e+5)
# gamma for MDP
GAMMA = 0.99

'''Training settings'''
# check GPU usage
USE_GPU = torch.cuda.is_available()
print('USE GPU: '+str(USE_GPU))
# mini-batch size
BATCH_SIZE = 64
# the multiple of tiling states, which helps us to adjust the relative importance 
# of the velocity vector against the image
N_TILE = 20

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Actor(nn.Module):
    def __init__(self, channel_num, state_dim, action_dim, max_action):
        """
        :param channel_num -> the number of channels for the image;
        :param state_dim -> the dimension of the velocity vector;
        :param action_dim -> the dimension of the action space;
        :param max_action -> the maximum value for each dimension of the action space;
        """
        super(Actor, self).__init__() 
        self.max_action = max_action
        
        self.feature_extraction = nn.Sequential(
            # Conv2d(输入channels, 输出channels, kernel_size, stride)
            # 8 -> 10
            nn.Conv2d(channel_num, 2, kernel_size=8, stride=2),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(2, 4, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        
        # Remove velocity information.
        self.fc_0 = nn.Linear(8 * 8 * 4 + N_TILE * state_dim, 400)
        # self.fc_0 = nn.Linear(8 * 8 * 10, 400)
        self.fc_1 = nn.Linear(400, 300)
        # action value
        self.fc_2 = nn.Linear(300, action_dim) 
        
        # 初始化参数值    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            
    def forward(self, image, velocity):
        """
        :param image -> image;
        :param velocity -> velocity vector;
        :return action
        """
        # x.size(0) : minibatch size
        mb_size = image.size(0)
        x = self.feature_extraction(image / 255.0) # (m, 9 * 9 * 10)
        x = x.view(x.size(0), -1)
        velocity = velocity.view(velocity.size(0), -1)
        # print("velocity: ", velocity)
        velocity = torch.tile(velocity, (1, N_TILE))
        # Remove the state information.
        x = torch.cat((x, velocity), 1)
        x = F.relu(self.fc_0(x))
        x = F.relu(self.fc_1(x))
        action = self.max_action * torch.tanh(self.fc_2(x))

        return action

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        
        
class Critic(nn.Module):
    def __init__(self, channel_num, state_dim, action_dim):
        """
        :param channel_num -> the number of channels for the image;
        :param state_dim -> the dimension of the velocity vector;
        :param action_dim -> the dimension of the action space;
        """
        super(Critic, self).__init__()
        
        self.feature_extraction = nn.Sequential(
            # Conv2d(输入channels, 输出channels, kernel_size, stride)
            # 8 -> 10
            nn.Conv2d(channel_num, 2, kernel_size=8, stride=2),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(2, 4, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        
        # Remove velocity information.
        self.fc_0 = nn.Linear(8 * 8 * 4 + N_TILE * state_dim + action_dim, 400)
        # self.fc_0 = nn.Linear(8 * 8 * 10, 400)  
        self.fc_1 = nn.Linear(400, 300)
        # action value
        self.fc_2 = nn.Linear(300, 1) 
        
        # 初始化参数值    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            
    def forward(self, image, velocity, action):
        """
        :param image -> image;
        :param state -> velocity vection;
        :param: action -> action;
        :return value -> q_value;
        """
        # x.size(0) : minibatch size
        mb_size = image.size(0)
        x = self.feature_extraction(image / 255.0) # (m, 9 * 9 * 10)
        x = x.view(x.size(0), -1)
        velocity = velocity.view(velocity.size(0), -1)
        velocity = torch.tile(velocity, (1, N_TILE))
        action = action.view(action.size(0), -1)
        # Remove the state information.
        x = torch.cat((x, velocity, action), 1)
        x = F.relu(self.fc_0(x))
        x = F.relu(self.fc_1(x))
        value = self.fc_2(x)

        return value

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DDPG(object):
    def __init__(self, channel_num, state_dim, action_dim, max_action):
        self.max_action = max_action 
        
        self.actor = Actor(channel_num, state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(channel_num, state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        # define optimizer of actor 7e-5
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=5e-5)
        
        self.critic = Critic(channel_num, state_dim, action_dim).to(device)
        self.critic_target = Critic(channel_num, state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # define optimizer of critic 9e-5
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=8e-5)
        
        # simulator step counter
        self.memory_counter = 0
        # target network step counter
        self.learn_step_counter = 0
        # ceate the replay buffer
        self.replay_buffer = ReplayBufferImage(MEMORY_CAPACITY)
        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(action_dim)
        
    def update_target(self, target, pred, update_rate):
        # update target network parameters using predcition network
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_(
                (1.0 - update_rate) * target_param.data + update_rate * pred_param.data
                )

    def choose_action(self, s, evaluate=False):
        # x:state
        image = np.stack([item[0] for item in s])
        state = np.stack([item[1] for item in s])
        image = torch.FloatTensor(image).to(device)
        state = torch.FloatTensor(state).to(device)
        
        if not evaluate:
            action = self.actor(image, state).cpu().data.numpy().squeeze()
            action = action + self.exploration_noise.noise()
            action = np.clip(action, -self.max_action, self.max_action)
        elif evaluate:
            action = self.actor(image, state).cpu().data.numpy().squeeze()
        return action

    def store_transition(self, s, a, r, s_, done, info):
        if done:
            self.exploration_noise.reset()
        for i in range(len(s)):
            self.memory_counter += 1
            self.replay_buffer.add(s[i], a[i], r[i], s_[i], float(done[i]), info)

    def learn(self):
        self.learn_step_counter += 1
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.update_target(self.actor_target, self.actor, 1e-2)
            self.update_target(self.critic_target, self.critic, 1e-2)
    
        b_i, b_s, b_a, b_r, b_i_, b_s_, b_d, b_info = self.replay_buffer.sample(BATCH_SIZE)
        # b_w, b_idxes = np.ones_like(b_r), None
        
        image = torch.FloatTensor(b_i).to(device) 
        velocity = torch.FloatTensor(b_s).to(device)
        action = torch.LongTensor(b_a).to(device)
        reward = torch.FloatTensor(b_r).reshape(-1, 1).to(device)
        next_image = torch.FloatTensor(b_i_).to(device)
        next_velocity = torch.FloatTensor(b_s_).to(device)
        done = torch.FloatTensor(b_d).reshape(-1, 1).to(device)
        info = b_info
        
        # compute the target Q value
        next_target_a = self.actor_target(next_image, next_velocity)
        target_next_Q = self.critic_target(next_image, next_velocity, next_target_a)
        target_Q = reward + GAMMA * (1 - done) * target_next_Q.detach()
        
        # compute the current Q estimate
        current_Q = self.critic(image, velocity, action)
        
        # compute the critic loss 
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        # optimize the critic 
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # compute the actor loss 
        expected_a = self.actor(image, velocity)
        # actor_loss = -self.critic(image, velocity, expected_a).mean()
        
        # the current Q estimate of the expected action
        actor_loss = -self.critic(image, velocity, expected_a).mean()
        
        # optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return critic_loss
    
    def save_model(self, algo, model_path):
        # save prediction network and target network
        torch.save(algo, model_path)

    def load_model(self, model_path):
        # load prediction network and target network
        model = torch.load(model_path)
        return model
        
    def save_buffer(self, buffer_path):
        self.replay_buffer.save_data(buffer_path)
        print("Successfully save buffer!")

    def load_buffer(self, buffer_path):
        # load data from the pkl file
        self.replay_buffer.read_list(buffer_path)
