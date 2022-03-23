import numpy as np


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUProcess(object):

    def __init__(self, n_actions, theta=0.15, mu=0, sigma=0.2, ):
        self.n_actions = n_actions
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.current_value = np.ones(self.n_actions) * self.mu

    def reset(self, sigma=0):
        self.current_value = np.ones(self.n_actions) * self.mu
        if sigma != 0:
            self.sigma = sigma

    def noise(self):
        x = self.current_value
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.current_value = x + dx
        return self.current_value
