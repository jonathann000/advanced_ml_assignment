import argparse
import gymnasium as gym
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import riverswim

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent_esarsa.py")
parser.add_argument("--env", type=str, help="Environment", default="riverswim:RiverSwim")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)
n_episodes = 100000

def run_agent(q_init=0.1, n_episodes=n_episodes):
    try:
        env = riverswim.RiverSwim()  # Local RiverSwim environment
        print("Loaded local RiverSwim environment")
    except Exception as e:
        print("Error loading local RiverSwim environment:", str(e))
        return []
    
    rewards_per_step = []  # Track rewards per step (instead of accumulated)
    action_dim = env.action_space.n
    state_dim = env.observation_space.n

    agent = agentfile.Agent(state_dim, action_dim, q_init=q_init)

    observation = env.reset()
    episode_count = 0
    while episode_count < n_episodes:  # Fix the number of episodes
        action = agent.act(observation)
        observation, reward, done, truncated, info = env.step(action)
        rewards_per_step.append(reward)  # Track reward for each step
        agent.observe(observation, reward, done)
        episode_count += 1  # Increment episode count after each step since RiverSwim is continuous

    env.close()
    return rewards_per_step

def plot_mean_rewards(all_rewards, r_length, n_episodes, q_init):
    """
    Plots the mean rewards per step and confidence intervals for a single Q-value initialization across multiple runs.
    :param all_rewards: List of reward arrays for each run of the same Q-value initialization
    :param r_length: Window size for the moving average
    :param n_episodes: Total number of episodes (steps) in each run
    """
    # Convert all_rewards to a NumPy array for easier manipulation
    all_rewards = np.array(all_rewards)
    
    # Calculate moving averages (using convolution) for each run
    moving_avg_rewards = np.array([np.convolve(r, np.ones(r_length)/r_length, mode='valid') for r in all_rewards])

    # Calculate mean and confidence interval across runs for each step
    mean_rewards = np.mean(moving_avg_rewards, axis=0)  # Mean reward per step, averaged over runs
    confidence_intervals = stats.sem(moving_avg_rewards, axis=0) * stats.t.ppf((1 + 0.95) / 2, moving_avg_rewards.shape[0] - 1)

    # Generate the steps array (taking into account the reduced number after moving average)
    steps = np.arange(n_episodes - r_length + 1)

    # Plot the mean and confidence intervals
    plt.plot(steps, mean_rewards, label=f'Q-init: {q_init}')  # Plot the mean rewards
    plt.fill_between(steps, mean_rewards - confidence_intervals, mean_rewards + confidence_intervals, alpha=0.2)

    plt.xlabel('Steps')
    plt.ylabel('Mean Reward per Step')
    plt.title('Mean Reward per Step with Confidence Intervals')
    plt.legend()
    plt.show()

# Number of runs for averaging
n_runs = 5
q_init_value = 0.2  # Set your Q-init value here
n_episodes = 100000  # Fix the number of episodes (steps) for all runs
r_length = 500  # Moving average window size

# Perform multiple runs of the agent with the same Q-value initialization
all_rewards = []
for _ in range(n_runs):
    rewards = run_agent(q_init=q_init_value, n_episodes=n_episodes)
    all_rewards.append(rewards)

# Plot the results for the single Q-init value
plot_mean_rewards(all_rewards, r_length=r_length, n_episodes=n_episodes, q_init=q_init_value)
