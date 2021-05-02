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
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

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


n_episodes=500
max_t=1000
agent = DDPG_Agent(**config)
scores = []                        # list containing scores from each episode
scores_window = deque(maxlen=100)  # last 100 scores
solved = False

for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=True)[brain_name]         # reset the environment    
    states = env_info.vector_observations[0]                  # get the current state (for each agent)
    score = 0                                                 # initialize the score (for each agent)
    agent.reset()
    for t in range(max_t):
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]              # send all actions to tne environment
        next_states = env_info.vector_observations[0]         # get next state (for each agent)
        rewards = env_info.rewards[0]                         # get reward (for each agent)
        dones = env_info.local_done[0]                        # see if episode finished
        score += rewards                                      # update the score (for each agent)

        agent.step(states, actions, rewards, next_states, dones) # step the agent

        states = next_states                               # roll over states to next time step
        if dones:                                          # exit loop if episode finished
            print("reached it! breaking!")
            break

    scores_window.append(score)       # save most recent score
    scores.append(score)               # save most recent score

    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        # Saving        
        agent.save()
        filename = 'scores'
        outfile = open(filename,'wb')
        pickle.dump(scores,outfile)
        outfile.close()

        if solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            agent.save()
            break
    if np.mean(scores_window)>=50:
        solved = True