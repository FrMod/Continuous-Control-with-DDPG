import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

import matplotlib.pylab as plt
import copy

import numpy as np

from Critic import DDPG_Critic
from Actor import DDPG_Actor
from ReplayBuffer import ReplayBuffer
from OrnsteinUhlenbeckNoise import OrnsteinUhlenbeckNoise

class DDPG_Agent(object):
    def __init__(self, **config):
        self.seed = config["seed"]
        self.tau = config["tau"]
        self.gamma = config["gamma"]
        self.action_size = config["action_size"]
        self.input_size = config["input_size"]
        self.batch_size = config["batch_size"]
        
        self.noise = OrnsteinUhlenbeckNoise(np.zeros(self.action_size), theta=config["noise_theta"], sigma=config["noise_sigma"])
        self.memory = ReplayBuffer(config["memory_size"], self.input_size, self.action_size, seed=self.seed)
        
        self.critic = DDPG_Critic(critic_lr=config["critic_lr"],
                                  input_size=self.input_size,    
                                  action_size=self.action_size,
                                  name="critic",
                                  weight_decay=config["weight_decay"],
                                  hidden_dim=config["critic_hidden_dim"],
                                  seed=self.seed)
        
        self.actor = DDPG_Actor(actor_lr=config["actor_lr"],
                                  input_size=self.input_size,    
                                  action_size=self.action_size,
                                  name="actor",    
                                  hidden_dim=config["actor_hidden_dim"],
                                  seed=self.seed+1)
        
        self.critic_target = DDPG_Critic(critic_lr=config["critic_lr"],
                                  input_size=self.input_size,    
                                  action_size=self.action_size,
                                  name="critic_target",   
                                  weight_decay=config["weight_decay"],
                                  hidden_dim=config["critic_hidden_dim"],
                                  seed=self.seed)

        self.actor_target = DDPG_Actor(actor_lr=config["actor_lr"],
                                  input_size=self.input_size,    
                                  action_size=self.action_size,
                                  name="actor_target",    
                                  hidden_dim=config["actor_hidden_dim"],
                                  seed=self.seed+1)

        print("===================Actor network================================")
        print(self.actor)
        print("================================================================")
        
        print()
        print("===================Critic network===============================")
        print(self.critic)
        print("================================================================")
        
        self.soft_update(self.critic, self.critic_target, tau=1)
        self.soft_update(self.actor, self.actor_target, tau=1)
    
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
        self.update_every = config["update_every"]
        
    def act(self, state):
        self.actor.eval()
        state = torch.from_numpy(state).float().to(self.actor.device)
        n = torch.from_numpy(self.noise.sample()).float().to(self.actor.device)
        action_values = self.actor.forward(state) + n
        self.actor.train()
        action_values = action_values.cpu().detach().numpy()
        return action_values #np.clip(action_values,-1,1)  # clip output 

    
    def step(self, state, action, reward, next_state, done):
        self.memory.add_memory(state, action, next_state, reward, done)
        self.learn()
        if self.t_step % self.update_every == 0:
            #update the weight
            self.soft_update(self.critic, self.critic_target, tau=self.tau)
            self.soft_update(self.actor, self.actor_target, tau=self.tau)

        self.t_step +=1
        
    def learn(self):
        if self.memory.index < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample_memory(self.batch_size)
        
        rewards = torch.tensor(reward, dtype=torch.float).view(self.batch_size,1).to(self.critic.device)
        dones = torch.tensor(done).view(self.batch_size,1).to(self.critic.device)
        next_states = torch.tensor(next_state, dtype=torch.float).to(self.critic.device)
        actions = torch.tensor(action, dtype=torch.float).to(self.critic.device)
        states = torch.tensor(state, dtype=torch.float).to(self.critic.device)    
    
        self.actor_target.eval()
        self.critic_target.eval()
        self.critic.eval()
        
        action_target = self.actor_target.forward(next_states)
        Q_targets_next = self.critic_target.forward(next_states, action_target)
        Q_expected = self.critic.forward(states, actions)
        
        # Compute Q targets for current states (y_i)
        
        Q_targets = rewards + (self.gamma * Q_targets_next * dones)      

        # update the critic
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(Q_expected,Q_targets)
        critic_loss.backward()
        self.critic.optimizer.step()
        
        # update the actor
        self.critic.eval()
        self.actor.optimizer.zero_grad()
        actions_pred = self.actor.forward(states)
        self.actor.train()
        
        actor_loss = -self.critic(states, actions_pred)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()    
        
    def soft_update(self, network, target_network, tau):
        for target_param, local_param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
            
    def reset(self):
        self.noise.reset()
        
    def save(self):
        print("...saving parameters...")
        self.critic.save()
        self.actor.save()
        self.critic_target.save()
        self.actor_target.save()
    
    def load(self, path):
        print("...loading parameters...")
        self.critic.load(path)
        self.actor.load(path)
        self.critic_target.load(path)
        self.actor_target.load(path)