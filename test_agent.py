from unityagents import UnityEnvironment
import numpy as np
from Agent import DDPG_Agent
from collections import deque
import pickle
import os

dir = ""    # PATH TO SIMULATOR EXECUTABLE
env = UnityEnvironment(file_name=dir)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)

# size of each action
action_size = brain.vector_action_space_size

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]

config = {"noise_theta":0.15,
          "noise_sigma":0.2,
          "memory_size":int(1e6), 
          "batch_size":128,
          "input_size":state_size, 
          "action_size":action_size,
          "seed":54,
          "actor_hidden_dim":(400,300),
          "critic_hidden_dim":(400,300),
          "critic_lr": 0.00025,
          "actor_lr":0.000025,
          "tau":0.001,
          "gamma":0.99,      
          "weight_decay":0,
          "update_every":1,
         }

agent = DDPG_Agent(**config)
curent_dir = os.getcwd()
agent.load(curent_dir+"/TrainedAgent")
scores = []                        # list containing scores from each episode

env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
states = env_info.vector_observations                  # get the current state (for each agent)

while True:
    actions = agent.act(states)
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
