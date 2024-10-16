import argparse
import gymnasium as gym
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent_dq.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)

def run_agent():
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
        action = agent.act(observation)
        observation, reward, done, truncated, info = env.step(action)
        total_rewards_in_episode += reward
        agent.observe(observation, reward, done)

        if done:
            rewards_per_episode.append(total_rewards_in_episode)
            total_rewards_in_episode = 0
            observation, info = env.reset()

    env.close()
    return rewards_per_episode

# Number of runs for averaging
n_runs = 5
all_rewards = []

# Perform 5 runs of the agent and store the results
for i in range(n_runs):
    rewards = run_agent()
    all_rewards.append(rewards)

# Ensure all reward arrays have the same length by padding with zeros (or NaNs)
max_episodes = max(len(r) for r in all_rewards)
all_rewards_padded = np.array([np.pad(r, (0, max_episodes - len(r)), constant_values=np.nan) for r in all_rewards])

# Calculate moving averages and 95% confidence intervals
r_length = 500
moving_avg_rewards = np.array([np.convolve(r, np.ones(r_length)/r_length, mode='valid') for r in all_rewards_padded])

# Calculate mean and 95% confidence intervals
mean_moving_avg = np.nanmean(moving_avg_rewards, axis=0)
std_moving_avg = np.nanstd(moving_avg_rewards, axis=0)
conf_interval = stats.norm.interval(0.95, loc=mean_moving_avg, scale=std_moving_avg/np.sqrt(n_runs))

# Plot the results
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title("Moving Average of Rewards with 95% Confidence Interval")
ax.plot(mean_moving_avg, label="Mean Moving Average", color="blue")
ax.fill_between(
    range(len(mean_moving_avg)),
    conf_interval[0],
    conf_interval[1],
    color="lightblue",
    alpha=0.5,
    label="95% Confidence Interval"
)
ax.set_xlabel("Episode")
ax.set_ylabel("Rewards")
ax.legend()
plt.tight_layout()
plt.show()
