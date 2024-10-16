import argparse
import gymnasium as gym
import importlib.util

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent_q.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)

try:
    env = gym.make(args.env, is_slippery=False)
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
    # env.render()
    action = agent.act(observation) 
    observation, reward, done, truncated, info = env.step(action)
    total_rewards_in_episode += reward
    agent.observe(observation, reward, done)
    
    if done:
        rewards_per_episode.append(total_rewards_in_episode)
        total_rewards_in_episode = 0
        observation, info = env.reset() 

env.close()
print(f'accumulated rewards: {sum(rewards_per_episode)}')
#print(f'q-table: \n {(agent.q1_values+agent.q2_values) / 2}')
print(f'q-table: \n {agent.q_values}')
#print(f'number of episodes: \n {len(rewards_per_episode)}')

""" Plots for rewards and training error """
import numpy as np
import matplotlib.pyplot as plt

r_length = 500
# avg to make plots more visibly clear
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

