import numpy as np

class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space, alpha=0.05, epsilon=0.05, gamma=0.95):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q1_values = np.full((state_space, action_space), 0.5)
        self.q2_values = np.full((state_space, action_space), 0.5)

        self.prev_observation = None
        self.prev_action = None

        self.stupid_flag = False
        self.training_error = []

    def observe(self, observation, reward, done):
        if np.random.random() < 0.5:
            td = reward + (
                (not done) * self.gamma * self.q2_values[observation, np.argmax(self.q1_values[observation])]
                ) - self.q1_values[self.prev_observation, self.prev_action]
            self.q1_values[self.prev_observation, self.prev_action] += self.alpha * td

        else:
            td = reward + (
                (not done) * self.gamma * self.q1_values[observation, np.argmax(self.q2_values[observation])]
            ) - self.q2_values[self.prev_observation, self.prev_action]
            self.q2_values[self.prev_observation, self.prev_action] += self.alpha * td
        
        self.prev_observation = observation
        self.training_error.append(td)


    def act(self, observation):
        if not self.stupid_flag:
            observation = observation[0]
            self.stupid_flag = True

        combined_q_value = (self.q1_values[observation] + self.q2_values[observation])

        if np.random.random() < self.epsilon:
            self.prev_action = np.random.choice(self.action_space)
        
        else:
            self.prev_action = np.argmax(combined_q_value)
        
        self.prev_observation = observation

        return self.prev_action