import numpy as np
import copy

class OrnsteinUhlenbeckNoise(object):
    "Class for Ornstein–Uhlenbeck process"
    
    def __init__(self, mu, seed=123, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.reset()
        
    def reset(self):
        self.state = copy.copy(self.mu)
        
    def sample(self):
        x = self.state
        dx = self.theta*(self.mu - x) + self.sigma*np.random.normal(size=self.mu.shape)
        self.state = x + dx
        return self.state