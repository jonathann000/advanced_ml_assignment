import numpy as np

class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space, alpha=0.1, epsilon=0.05, gamma=0.95, q_init=.2):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_init = q_init
        self.q1_values = np.full((state_space, action_space), self.q_init)
        self.q2_values = np.full((state_space, action_space), self.q_init)

        self.prev_observation = None
        self.prev_action = None

        self.stupid_flag = False
        self.training_error = []

    def observe(self, observation, reward, done):
        if np.random.random() < 0.5:
            td = (
                reward + (
                (not done) * self.gamma * self.q2_values[observation, np.argmax(self.q1_values[observation])]
                ) - self.q1_values[self.prev_observation, self.prev_action]
                )
            self.q1_values[self.prev_observation, self.prev_action] += self.alpha * td

        else:
            td = (
                reward + (
                (not done) * self.gamma * self.q1_values[observation, np.argmax(self.q2_values[observation])]
                ) - self.q2_values[self.prev_observation, self.prev_action]
                )
            self.q2_values[self.prev_observation, self.prev_action] += self.alpha * td
        
        self.prev_observation = observation
        self.training_error.append(td)


    def act(self, observation):
        if not self.stupid_flag:
            observation = observation[0]
            self.stupid_flag = True

        combined_q_value = (self.q1_values[observation] + self.q2_values[observation]) / 2

        if np.random.random() > self.epsilon and np.max(combined_q_value) != self.q_init:
            action = np.argmax(combined_q_value)
        else:
            action = np.random.choice(self.action_space)
        
        self.prev_observation = observation
        self.prev_action = action

        return action