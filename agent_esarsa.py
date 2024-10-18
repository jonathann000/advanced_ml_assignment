import numpy as np

class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space, alpha=0.1, epsilon=0.05, gamma=0.95, q_init = 0.1):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_init = q_init
        self.q_values = np.full((state_space, action_space), self.q_init) # TODO: test different initialisation values
        self.prev_observation = None
        self.prev_action = None

        self.stupid_flag = False
        self.training_error = []

    def observe(self, observation, reward, done):
        expected_q = 0
        greedy_actions = 0
        q_max = np.max(self.q_values[observation])
        for i in range(self.action_space):
            if self.q_values[observation, i] == q_max:
                greedy_actions += 1

        prob_non_greedy = self.epsilon / self.action_space
        prob_greedy = ((1 - self.epsilon) / greedy_actions) + prob_non_greedy
 
        for i in range(self.action_space):
            if self.q_values[observation, i] == q_max:
                expected_q += prob_greedy * q_max
            else:
                expected_q += prob_non_greedy * self.q_values[observation, i]
        
        td = reward + ((not done) * self.gamma * expected_q) - self.q_values[self.prev_observation, self.prev_action]

        self.q_values[self.prev_observation, self.prev_action] += self.alpha * td 

        self.prev_observation = observation
        self.training_error.append(td)


    def act(self, observation):
        if not self.stupid_flag:
            observation = observation[0]
            self.stupid_flag = True

        if np.random.random() > self.epsilon and np.max(self.q_values[observation]) != self.q_init:
            action = np.argmax(self.q_values[observation])
        else:
            action = np.random.choice(self.action_space)
        
        self.prev_observation = observation
        self.prev_action = action

        return self.prev_action