# Define a rollout buffer for policy gradient method. 

class RolloutBuffer(object):
    def __init__(self):
        self.actions = []
        self.images = []
        self.states = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.images[:]
        del self.rewards[:]
        del self.is_terminals[:]
        
        
class ParallelRolloutBuffer(object):
    def __init__(self, n_envs):
        self.n_envs = n_envs 
        self.buffers = [RolloutBuffer() for _ in range(int(n_envs))]
    
    def add(self, state, action, reward, is_terminal):
        for idx in range(int(self.n_envs)):
            self.buffers[idx].images.append(state[idx][0])
            self.buffers[idx].states.append(state[idx][1])
            self.buffers[idx].actions.append(action[idx])
            self.buffers[idx].rewards.append(reward[idx])
            self.buffers[idx].is_terminals.append(is_terminal[idx])

    def to_simple_buffer(self):
        simple_buffer = RolloutBuffer()
        for idx in range(int(self.n_envs)):
            simple_buffer.images += self.buffers[idx].images 
            simple_buffer.states += self.buffers[idx].states 
            simple_buffer.actions += self.buffers[idx].actions 
            simple_buffer.rewards += self.buffers[idx].rewards 
            simple_buffer.is_terminals += self.buffers[idx].is_terminals 
        
        return simple_buffer
    
    def clear(self):
        for idx in range(int(self.n_envs)):
            self.buffers[idx].clear()