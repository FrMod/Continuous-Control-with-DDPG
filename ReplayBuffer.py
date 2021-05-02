import numpy as np

class ReplayBuffer(object):
    """Replay Buffer class
       Based on implementation by Phil Tabor of DDPG architecture
    """
    
    def __init__(self, memory_size, input_size, action_size, seed=123):
        self.memory_size = memory_size
        self.index = 0
        self.seed = np.random.seed(seed)
        
        #fields
        self.state = np.zeros((memory_size, input_size))
        self.action = np.zeros((memory_size, action_size))
        self.reward = np.zeros(memory_size)
        self.next_state = np.zeros((memory_size, input_size))
        self.done = np.zeros(memory_size,dtype=np.float32)
        
    def add_memory(self, state, action, next_state, reward, done_):
        indx = self.index % self.memory_size # Circular buffer
        
        self.state[indx] = state
        self.action[indx] = action
        self.reward[indx] = reward
        self.next_state[indx] = next_state
        self.done[indx] = 1-done_ 
        
        self.index = indx + 1
        
    def sample_memory(self, batch_size):
        max_index = min(self.index, self.memory_size)
        batch_index = np.random.choice(max_index, batch_size) # uniform sampling
        
        state = self.state[batch_index]
        action = self.action[batch_index]
        reward = self.reward[batch_index]
        next_state = self.next_state[batch_index]
        done = self.done[batch_index]
        
        return state, action, reward, next_state, done