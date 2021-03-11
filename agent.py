import numpy as np

class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space

    def observe(self, observation, reward, done):
        pass
    def act(self, observation):
        return np.random.randint(self.action_space)