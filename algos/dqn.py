###########################################################################################
# Implementation of Deep Q-Learning Networks (DQN)
# Paper: https://www.nature.com/articles/nature14236
# Reference: https://github.com/Kchu/DeepRL_PyTorch
###########################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .replay_memory import ReplayBufferImage


'''DQN settings'''
# sequential images to define state
STATE_LEN = 4
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
# learning rage
LR = 2e-4
# the number of actions 
N_ACTIONS = 9
# the dimension of states
N_STATE = 4
# the multiple of tiling states 
N_TILE = 20


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.feature_extraction = nn.Sequential(
        	# Conv2d(输入channels, 输出channels, kernel_size, stride)
            nn.Conv2d(STATE_LEN, 8, kernel_size=8, stride=2),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 10, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        # Remove velocity information.
        # self.fc_0 = nn.Linear(8 * 8 * 10 + N_TILE * N_STATE, 512)
        self.fc_0 = nn.Linear(8 * 8 * 10, 512)
        
        self.fc_1 = nn.Linear(512, 512)
           
        # action value
        self.fc_q = nn.Linear(512, N_ACTIONS) 
        
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
            
    def forward(self, x, state):
        # x.size(0) : minibatch size
        mb_size = x.size(0)
        x = self.feature_extraction(x / 255.0) # (m, 9 * 9 * 10)
        x = x.view(x.size(0), -1)
        state = state.view(state.size(0), -1)
        state = torch.tile(state, (1, N_TILE))
        # Remove the state information.
        # x = torch.cat((x, state), 1)
        x = F.relu(self.fc_0(x))
        x = F.relu(self.fc_1(x))
        action_value = self.fc_q(x)

        return action_value

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        

class DQN(object):
    def __init__(self):
        self.pred_net, self.target_net = ConvNet(), ConvNet()
        # sync evac target
        self.update_target(self.target_net, self.pred_net, 1.0)
        # use gpu
        if USE_GPU:
            self.pred_net.cuda()
            self.target_net.cuda()
            
        # simulator step counter
        self.memory_counter = 0
        # target network step counter
        self.learn_step_counter = 0
        # loss function
        self.loss_function = nn.MSELoss()
        # ceate the replay buffer
        self.replay_buffer = ReplayBufferImage(MEMORY_CAPACITY)
        
        # define optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=LR)
        
    def update_target(self, target, pred, update_rate):
        # update target network parameters using predcition network
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_((1.0 - update_rate) \
                                    * target_param.data + update_rate * pred_param.data)

    def choose_action(self, s, epsilon):
    	# x:state
        image = np.stack([item[0] for item in s])
        state = np.stack([item[1] for item in s])
        image = torch.FloatTensor(image)
        state = torch.FloatTensor(state)
        # print(x.shape)
        if USE_GPU:
            image = image.cuda()
            state = state.cuda()
        # epsilon-greedy策略
        if np.random.uniform() >= epsilon:
            # greedy case
            action_value = self.pred_net(image, state) 	# (N_ENVS, N_ACTIONS, N_QUANT)
            action = torch.argmax(action_value, dim=1).data.cpu().numpy()
        else:
            # random exploration case
            action = np.random.randint(0, N_ACTIONS, (image.size(0)))
        return action

    def store_transition(self, s, a, r, s_, done):
        for i in range(len(s)):
            self.memory_counter += 1
            self.replay_buffer.add(s[i], a[i], r[i], s_[i], float(done[i]))

    def learn(self):
        self.learn_step_counter += 1
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.update_target(self.target_net, self.pred_net, 1e-2)
    
        b_i, b_s, b_a, b_r, b_i_, b_s_, b_d = self.replay_buffer.sample(BATCH_SIZE)
        # b_w, b_idxes = np.ones_like(b_r), None
        
        b_i = torch.FloatTensor(b_i)    
        b_s = torch.FloatTensor(b_s)
        b_a = torch.LongTensor(b_a)
        b_r = torch.FloatTensor(b_r)
        b_i_ = torch.FloatTensor(b_i_)
        b_s_ = torch.FloatTensor(b_s_)
        b_d = torch.FloatTensor(b_d)

        if USE_GPU:
            b_i, b_s, b_a, b_r, b_i_, b_s_, b_d = b_i.cuda(), b_s.cuda(), b_a.cuda(), \
                b_r.cuda(), b_i_.cuda(), b_s_.cuda(), b_d.cuda()

        # action value for current state 
        q_eval = self.pred_net(b_i, b_s) 	
        mb_size = q_eval.size(0)
        q_eval = torch.stack([q_eval[i][b_a[i]] for i in range(mb_size)])

        # optimal action value for current state 
        q_next = self.target_net(b_i_, b_s_) 				
        # best_actions = q_next.argmax(dim=1) 		
        # q_next = torch.stack([q_next[i][best_actions[i]] for i in range(mb_size)])
        q_next = torch.max(q_next, -1)[0]
        q_target = b_r + GAMMA * (1. - b_d) * q_next
        q_target = q_target.detach()

        # loss
        loss = self.loss_function(q_eval, q_target)
        
        # backprop loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
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