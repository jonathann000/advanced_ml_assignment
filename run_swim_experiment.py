import argparse
import gymnasium as gym
import importlib.util
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent_esarsa.py")
parser.add_argument("--env", type=str, help="Environment", default="riverswim:RiverSwim")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)


try:
    env = gym.make(args.env)
    print("Loaded ", args.env)
except:
    file_name, env_name = args.env.split(":")
    gym.envs.register(
        id=env_name + "-v0",
        entry_point=args.env,
    )
    env = gym.make(env_name + "-v0")
    print("Loaded", args.env)

total_rewards_in_episode = 0
rewards_per_episode = []
action_dim = env.action_space.n
state_dim = env.observation_space.n

agent = agentfile.Agent(state_dim, action_dim)

observation = env.reset()
for _ in range(100000): 
    action = agent.act(observation) 
    observation, reward, done, truncated, info = env.step(action)
    total_rewards_in_episode += reward
    rewards_per_episode.append(total_rewards_in_episode)
    agent.observe(observation, reward, done)
    if done:
        rewards_per_episode.append(total_rewards_in_episode)
        total_rewards_in_episode = 0
        observation, info = env.reset() 

env.close()

def value_iteration(env, threshold=1e-6, gamma=0.95):
    V = np.zeros(env.observation_space.n)
    
    while True:
        delta = 0 
        for s in range(env.observation_space.n):
            v = V[s]  
            
            # Update V(s) using Bellman equation
            action_values = []
            for a in range(env.action_space.n):
                total = 0
                if a == 0:  # left
                    if s == 0:
                        reward = env.small
                        next_state = 0
                    else:
                        reward = 0
                        next_state = s - 1
                    total += reward + gamma * V[next_state]
                else:  # right
                    if s == 0:
                        reward = 0
                        next_state_prob = [0.4, 0.6]
                        next_state_options = [0, 1]
                    elif s < env.n - 1:
                        reward = 0
                        next_state_prob = [0.05, 0.6, 0.35]
                        next_state_options = [s - 1, s, s + 1]
                    else:  # s == n-1 
                        reward = env.large
                        next_state_prob = [0.4, 0.6]
                        next_state_options = [s - 1, s]
                        
                    for next_state, prob in zip(next_state_options, next_state_prob):
                        total += prob * (reward + gamma * V[next_state])
                
                action_values.append(total)
            
            V[s] = max(action_values)  # action with highest expected value
            delta = max(delta, abs(v - V[s])) 
 
        if delta < threshold:
            break

    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for s in range(env.observation_space.n):
        action_values = []
        for a in range(env.action_space.n):
            total = 0
            if a == 0:  # left
                if s == 0:
                    reward = env.small
                    next_state = 0
                else:
                    reward = 0
                    next_state = s - 1
                total += reward + gamma * V[next_state]
            else:  # right
                if s == 0:
                    reward = 0
                    next_state_prob = [0.4, 0.6]
                    next_state_options = [0, 1]
                elif s < env.n - 1:
                    reward = 0
                    next_state_prob = [0.05, 0.6, 0.35]
                    next_state_options = [s - 1, s, s + 1]
                else:  # s == n-1
                    reward = env.large
                    next_state_prob = [0.4, 0.6]
                    next_state_options = [s - 1, s]

                for next_state, prob in zip(next_state_options, next_state_prob):
                    total += prob * (reward + gamma * V[next_state])
                    
            action_values.append(total)
        
        best_action = np.argmax(action_values)
        policy[s, best_action] = 1.0 

    return policy, V

print(f'accummulated rewards and q-table: {rewards_per_episode[-1]} \n {(agent.q1_values+agent.q2_values) / 2}' )
#print(f'accummulated rewards and q-table: {rewards_per_episode[-1]} \n {agent.q_values}' )

""" Plots for rewards and training error """
import numpy as np
r_length = 500 
# avg to make plots more visibly clear
fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
rewards_moving_avg = (
    np.convolve(np.array(rewards_per_episode), np.ones(r_length), mode = 'valid')
    / r_length
)
axs[0].set_title("Rewards per step")
axs[0].plot(range(len(rewards_moving_avg)), rewards_moving_avg)
training_error_moving_avg = (
    np.convolve(np.array(agent.training_error), np.ones(r_length), mode = 'same')
    / r_length
)
axs[1].set_title("Training error")
axs[1].plot(range(len(training_error_moving_avg)), training_error_moving_avg)

plt.tight_layout()
plt.show()

q_table = (agent.q1_values + agent.q2_values) / 2
#q_table = agent.q_values
n_states = q_table.shape[0]

x = np.arange(n_states)  
y = np.ones(n_states) * 0.5  # Fixed y-value

dx = np.zeros_like(q_table[:, 0])
dy = np.zeros_like(q_table[:, 0])

# Left: dx negative, Right: dx positive
for i in range(n_states):
    best_action = np.argmax(q_table[i])
    
    if best_action == 0:  # Left
        dx[i] = -0.3
    elif best_action == 1:  # Right
        dx[i] = 0.3


plt.figure(figsize=(8, 2))
plt.quiver(x + 0.4, y, dx, dy, scale=1, scale_units='xy', angles='xy', color='r')

# Set equal spacing for arrows
plt.xlim(0, n_states)
plt.ylim(0, 1)

plt.xticks(np.arange(n_states) + 0.5, labels=np.arange(n_states))
plt.yticks([]) 

# Remove all borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

plt.gca().set_aspect('equal', adjustable='box')

plt.title('Q-table Preferred Actions (Left/Right)')

plt.show()

policy, V = value_iteration(env)
print("Optimal Value Function:")
print(V)
print("Optimal Policy:")
print(policy)

