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
n_episodes = 1000
n_runs = 3  # Number of runs to average per q_init

def run_agent(q_init=0.1, n_episodes=n_episodes):
    try:
        env = riverswim.RiverSwim()
        print("Loaded local RiverSwim environment")
    except Exception as e:
        print("Error loading local RiverSwim environment:", str(e))
        return []
    total_rewards_in_episode = 0
    rewards_per_episode = []
    action_dim = env.action_space.n
    state_dim = env.observation_space.n

    agent = agentfile.Agent(state_dim, action_dim, q_init=q_init)

    observation = env.reset()
    episode_count = 0
    while episode_count < n_episodes: 
        action = agent.act(observation)
        observation, reward, done, truncated, info = env.step(action)
        total_rewards_in_episode += reward
        agent.observe(observation, reward, done)
        rewards_per_episode.append(total_rewards_in_episode)
        episode_count += 1 # after each step since riverswim is continuous

    env.close()
    return rewards_per_episode

def plot_mean_rewards(all_rewards, q_inits, r_length):
    all_rewards = np.array(all_rewards)

    plt.figure(figsize=(10, 6))
    
    for i, q_init in enumerate(q_inits):
        moving_avg_rewards = np.array([np.convolve(r, np.ones(r_length)/r_length, mode='valid') for r in all_rewards[i]])

        # Calculate mean and confidence interval across runs for each episode
        mean_rewards = np.mean(moving_avg_rewards, axis=0) 
        confidence_intervals = stats.sem(moving_avg_rewards, axis=0) * stats.t.ppf((1 + 0.95) / 2, moving_avg_rewards.shape[0] - 1)

        # mean rewards with confidence intervals
        episodes = np.arange(n_episodes - r_length + 1)
        plt.plot(episodes, mean_rewards, label=f'Q-init: {q_init}')
        #plt.fill_between(episodes, mean_rewards - confidence_intervals, mean_rewards + confidence_intervals, alpha=0.2)

    plt.xlabel('Episodes')
    plt.ylabel('Accumulated Reward')
    plt.title('Accumulated Rewards for Different Q-value Initializations')
    plt.legend()
    plt.show()

q_inits = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
n_episodes = 100000
all_rewards = []

for q_init in q_inits:
    rewards_for_q_init = []
    for run in range(n_runs):
        rewards = run_agent(q_init=q_init, n_episodes=n_episodes)
        rewards_for_q_init.append(rewards)
    all_rewards.append(rewards_for_q_init)

plot_mean_rewards(all_rewards, q_inits, r_length=500)
