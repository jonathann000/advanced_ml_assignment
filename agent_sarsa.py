import numpy as np

class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space, alpha=0.1, epsilon=0.05, gamma=0.95, q_init = 0.2):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_init = q_init
        self.q_values = np.full((state_space, action_space), self.q_init)
        self.prev_observation = None
        self.prev_action = None

        self.stupid_flag = False # First observation is of other type
        # self.training_error = []

    def observe(self, observation, reward, done):
        if np.random.random() > self.epsilon and np.max(self.q_values[observation]) != self.q_init:
            action = np.argmax(self.q_values[observation])
        else:
            action = np.random.choice(self.action_space)

        td = reward + ((not done) * self.gamma * self.q_values[observation, action]) - self.q_values[self.prev_observation, self.prev_action]
        self.q_values[self.prev_observation, self.prev_action] += self.alpha * td

        self.prev_observation = observation
        self.prev_action = action
        # self.training_error.append(td)


    def act(self, observation):
        if not self.stupid_flag:
            observation = observation[0]
            self.stupid_flag = True

        # Exploit
        if np.random.random() > self.epsilon and np.max(self.q_values[observation]) != self.q_init:
            action = np.argmax(self.q_values[observation])
        # Explore
        else:
            action = np.random.choice(self.action_space)
        
        self.prev_observation = observation
        self.prev_action = action

        return self.prev_action