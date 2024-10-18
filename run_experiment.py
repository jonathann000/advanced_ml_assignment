import argparse
import gymnasium as gym
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent_esarsa.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)

try:
    env = gym.make(args.env, is_slippery=True)
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
n_episodes = 10000
episode_count = 0
while episode_count < n_episodes:
    action = agent.act(observation) 
    observation, reward, done, truncated, info = env.step(action)
    total_rewards_in_episode += reward
    agent.observe(observation, reward, done)
    
    if done:
        rewards_per_episode.append(total_rewards_in_episode)
        total_rewards_in_episode = 0
        observation, info = env.reset()
        episode_count += 1

env.close()
print(f'accumulated rewards: {sum(rewards_per_episode)}')
#print(f'q-table: \n {(agent.q1_values+agent.q2_values) / 2}')
print(f'q-table: \n {agent.q_values}')

def value_iteration(env, threshold=1e-6, gamma=0.95):
    V = np.zeros(env.observation_space.n)
    
    while True:
        delta = 0 
        for s in range(env.observation_space.n):
            v = V[s] 
            # Bellman equation
            V[s] = max([sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.action_space.n)])
            delta = max(delta, abs(v - V[s]))

        if delta < threshold:
            break
    
    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for s in range(env.observation_space.n):
        # select action that maximizes the value
        best_action = np.argmax([sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.action_space.n)])
        policy[s, best_action] = 1.0  # best action for state
    
    return policy, V


""" Plots for rewards and training error """
r_length = 500
# avg to make plots more clear
fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
rewards_moving_avg = (
    np.convolve(np.array(rewards_per_episode), np.ones(r_length), mode = 'valid')
    / r_length
)
axs[0].set_title("Rewards per episode")
axs[0].plot(range(len(rewards_moving_avg)), rewards_moving_avg)

training_error_moving_avg = (
    np.convolve(np.array(agent.training_error), np.ones(r_length), mode = 'same')
    / r_length
)
axs[1].set_title("Training error")
axs[1].plot(range(len(training_error_moving_avg)), training_error_moving_avg)

plt.tight_layout()
plt.show()

q_table = agent.q_values
#q_table = (agent.q1_values + agent.q2_values) / 2
actions = ['left', 'down', 'right', 'up']
n_states = q_table.shape[0]

# grid for visualization 
grid_size = int(np.sqrt(n_states))
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))

# center arrows
x = x + 0.5
y = y + 0.5

# Which way arrow is pointing
dx = np.zeros_like(q_table[:, 0]) 
dy = np.zeros_like(q_table[:, 0])

for i in range(n_states):
    if np.all(q_table[i] == agent.q_init):  # Terminal state
        dx[i] = 0
        dy[i] = 0
        continue
    best_action = np.argmax(q_table[i])
    
    if best_action == 0:  # Left
        dx[i] = -0.3
        dy[i] = 0
    elif best_action == 1:  # Down
        dx[i] = 0
        dy[i] = 0.3
    elif best_action == 2:  # Right
        dx[i] = 0.3
        dy[i] = 0
    elif best_action == 3:  # Up
        dx[i] = 0
        dy[i] = -0.3

plt.figure(figsize=(6, 6))
plt.quiver(x, y, dx.reshape(grid_size, grid_size), dy.reshape(grid_size, grid_size), 
           scale=1, scale_units='xy', angles='xy', color='r')

plt.xlim(0, grid_size)
plt.ylim(0, grid_size)
plt.gca().invert_yaxis()  # its upside down (:
plt.xticks(np.arange(grid_size))
plt.yticks(np.arange(grid_size))
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box') 
plt.title('Q-table Preferred Actions')
plt.show()

# Run value iteration to get optimal policy and value function
policy, V = value_iteration(env)

print("Optimal policy:")
print(np.argmax(policy, axis=1).reshape(grid_size, grid_size))
print("Optimal value function:")
print(V.reshape(grid_size, grid_size))





