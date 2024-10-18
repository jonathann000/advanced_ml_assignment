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
        env = riverswim.RiverSwim()  
        print("Loaded local RiverSwim environment")
    except Exception as e:
        print("Error loading local RiverSwim environment:", str(e))
        return []
    
    rewards_per_step = [] 
    action_dim = env.action_space.n
    state_dim = env.observation_space.n

    agent = agentfile.Agent(state_dim, action_dim, q_init=q_init)

    observation = env.reset()
    episode_count = 0
    while episode_count < n_episodes:  
        action = agent.act(observation)
        observation, reward, done, truncated, info = env.step(action)
        rewards_per_step.append(reward) 
        agent.observe(observation, reward, done)
        episode_count += 1  # after each step since riverswim is continuous

    env.close()
    return rewards_per_step

def plot_mean_rewards(all_rewards, r_length, n_episodes, q_init):

    all_rewards = np.array(all_rewards)
    moving_avg_rewards = np.array([np.convolve(r, np.ones(r_length)/r_length, mode='valid') for r in all_rewards])

    # Calculate mean and confidence interval across runs for each step
    mean_rewards = np.mean(moving_avg_rewards, axis=0) 
    confidence_intervals = stats.sem(moving_avg_rewards, axis=0) * stats.t.ppf((1 + 0.95) / 2, moving_avg_rewards.shape[0] - 1)

    # Generate steps array
    steps = np.arange(n_episodes - r_length + 1)

    # mean and confidence intervals
    plt.plot(steps, mean_rewards, label=f'Q-init: {q_init}')
    plt.fill_between(steps, mean_rewards - confidence_intervals, mean_rewards + confidence_intervals, alpha=0.2)

    plt.xlabel('Steps')
    plt.ylabel('Mean Reward per Step')
    plt.title('Mean Reward per Step with Confidence Intervals')
    plt.legend()
    plt.show()

n_runs = 5
q_init_value = 0.2  
n_episodes = 100000  
r_length = 500  

# multiple runs with same q init to get confidence average
all_rewards = []
for _ in range(n_runs):
    rewards = run_agent(q_init=q_init_value, n_episodes=n_episodes)
    all_rewards.append(rewards)

plot_mean_rewards(all_rewards, r_length=r_length, n_episodes=n_episodes, q_init=q_init_value)
