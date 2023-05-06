import torch 
import numpy as np
import time


def rollout(ppo_agent, env, max_ep_len, device, render=True, speedup=None):
    state = env.reset()
    current_ep_reward = 0
    step = 0

    observations = []
    actions = []
    action_logprobs = []
    dist_probs = []
    rewards = []
    com = []  # com may not be the same as state always.
    is_terminals = [] 
    goal = None

    for t in range(1, max_ep_len+1):
        com.append(state[0:2])
        state = torch.FloatTensor(state).to(device)
        observations.append(state)
        # select action with policy
        action, action_logprob, dist_prob = ppo_agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        # saving reward and is_terminals
        action = np.array([action])
        action = torch.FloatTensor(action).to(device)
        actions.append(action)
        action_logprobs.append(action_logprob)
        dist_probs.append(dist_prob)
        rewards.append(reward)
        is_terminals.append(done)

        step +=1
        state = next_state
        current_ep_reward += reward
        
        if render:
            env.render()
            # time_step = 0.05 
            # time.sleep(time_step / speedup)
        
        # break; if the episode is over
        if done:
            # goal = env.arrive_goal(state)
            goal = env.arrive_goal()
            break
        
    path = {
        "observations": observations,
        "actions": actions,
        "logprobs": action_logprobs,
        "distprobs": dist_probs,
        "rewards": rewards, 
        "is_terminals": is_terminals,
        "current_ep_reward": current_ep_reward,
        "time_step": step,
        "goal": goal,
        "com": com,
    }
    return path


class Sampler():
    def __init__(
        self, marl_frame, ppo_agent, env, max_ep_len, n_episodes, 
        device, use_mmd=False, render=True, speedup=None):
        self.env = env 
        self.marl_frame = marl_frame
        self.ppo_agent = ppo_agent 
        self.max_ep_len = max_ep_len 
        self.n_episodes = n_episodes 
        self.render = render 
        self.speedup = speedup 
        self.gamma = ppo_agent.gamma
        self.device = device
        self.use_mmd = use_mmd
    
    def sample_paths(self):
        paths = []
        for i in range(self.n_episodes):
            path = rollout(
                self.ppo_agent, self.env, self.max_ep_len, self.device, 
                self.render, self.speedup)
            paths.append(path)
        # time_step = 0
        # i = 0
        # while i < self.n_episodes or time_step < 2048:
        #     path = rollout(
        #         self.ppo_agent, self.env, self.max_ep_len, self.device, 
        #         self.render, self.speedup)
        #     paths.append(path)
        #     i += 1
        #     time_step += len(path["observations"])
        
        return paths

    def process_samples(self, paths, i_actor, goal_records):
        returns = []
        observations = [] 
        actions = []
        log_probs = [] 
        dist_probs = []
        is_terminals = []
        
        for idx, path in enumerate(paths):
            observations += path["observations"]
            actions += path["actions"]
            log_probs += path["logprobs"]
            dist_probs += path["distprobs"]
            is_terminals += path["is_terminals"]
            
            # Monte Carlo estimate of returns
            rewards = []
            discounted_reward = 0
            for reward in reversed(path["rewards"]):
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward) 
            path["returns"] = rewards
            returns += rewards 
                
        # Normalizing the rewards
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        rewards = (returns - returns.mean()) / (returns.std() + 1e-7) 
    
		sample_data = dict(
			observations=observations,
			actions=actions,
			log_probs=log_probs,
			dist_probs=dist_probs,
			rewards=rewards,
			is_terminals=is_terminals
		)
  
        return sample_data