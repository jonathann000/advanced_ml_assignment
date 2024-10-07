import numpy as np

class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space, alpha=0.1, epsilon=0.05, gamma=0.95):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_values = np.full((state_space, action_space), 1.0)
        self.prev_observation = None
        self.prev_action = None

        self.stupid_flag = False



    def observe(self, observation, reward, done):
        #Add your code here

        self.q_values[self.prev_observation, self.prev_action] += self.alpha * (reward + ((not done) * self.gamma * np.max(self.q_values[observation])) - self.q_values[self.prev_observation, self.prev_action])
        self.prev_observation = observation
        #print(self.q_values)
        
    def act(self, observation):
        if not self.stupid_flag:
            observation = observation[0]
            self.stupid_flag = True
        if np.random.rand() < self.epsilon:
            self.prev_action = np.random.choice(range(self.action_space))
        
        else:
            self.prev_action = np.argmax(self.q_values[observation])

        return self.prev_action