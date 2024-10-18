import argparse
import gymnasium as gym
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent_esarsa.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)
n_episodes = 1000
n_runs = 3  # Number of runs to average per q_init

def run_agent(q_init=0.1, n_episodes=n_episodes):
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

    agent = agentfile.Agent(state_dim, action_dim, q_init=q_init)

    observation = env.reset()
    episode_count = 0
    while episode_count < n_episodes:  # Fix the number of episodes
        action = agent.act(observation)
        observation, reward, done, truncated, info = env.step(action)
        total_rewards_in_episode += reward
        agent.observe(observation, reward, done)

        if done:
            rewards_per_episode.append(total_rewards_in_episode)
            total_rewards_in_episode = 0
            observation, info = env.reset()
            episode_count += 1  # Increment episode count after each reset
    env.close()
    return rewards_per_episode

def plot_mean_rewards(all_rewards, q_inits, r_length):
    """
    Plots the mean rewards for different Q-value initializations.
    :param all_rewards: List of reward arrays for each Q-value initialization
    :param q_inits: List of Q-value initialization values
    :param r_length: Window size for the moving average
    """
    # Convert all_rewards to a NumPy array for easier manipulation
    all_rewards = np.array(all_rewards)

    plt.figure(figsize=(10, 6))
    
    for i, q_init in enumerate(q_inits):
        # Calculate moving averages (using convolution) for each Q-init
        moving_avg_rewards = np.array([np.convolve(r, np.ones(r_length)/r_length, mode='valid') for r in all_rewards[i]])

        # Calculate mean and confidence interval across runs for each episode
        mean_rewards = np.mean(moving_avg_rewards, axis=0)  # Mean reward per episode, averaged over runs
        confidence_intervals = stats.sem(moving_avg_rewards, axis=0) * stats.t.ppf((1 + 0.95) / 2, moving_avg_rewards.shape[0] - 1)

        # Plot the mean rewards with confidence intervals
        episodes = np.arange(n_episodes - r_length + 1)
        plt.plot(episodes, mean_rewards, label=f'Q-init: {q_init}')
        #plt.fill_between(episodes, mean_rewards - confidence_intervals, mean_rewards + confidence_intervals, alpha=0.2)

    plt.xlabel('Episodes')
    plt.ylabel('Moving Average Reward')
    plt.title('Moving Average for Different Q-value Initializations')
    plt.legend()
    plt.show()

# List of Q-value initializations to test
q_inits = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]

# Perform 5 runs of the agent for each Q-value initialization
n_episodes = 10000  # Fix the number of episodes for all runs
all_rewards = []

for q_init in q_inits:
    rewards_for_q_init = []
    for run in range(n_runs):
        rewards = run_agent(q_init=q_init, n_episodes=n_episodes)
        rewards_for_q_init.append(rewards)
    all_rewards.append(rewards_for_q_init)

# Plot the mean rewards for different Q-value initializations
plot_mean_rewards(all_rewards, q_inits, r_length=500)
