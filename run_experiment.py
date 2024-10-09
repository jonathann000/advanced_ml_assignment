import argparse
import gymnasium as gym
import importlib.util
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
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


rewards = []
action_dim = env.action_space.n
state_dim = env.observation_space.n

agent = agentfile.Agent(state_dim, action_dim)

observation = env.reset()
for _ in range(1000000): 
    #env.render()
    action = agent.act(observation) 
    observation, reward, done, truncated, info = env.step(action)
    rewards.append(reward)
    agent.observe(observation, reward, done)
    
    if done:
        observation, info = env.reset() 

env.close()

# Plots Q-learning
import numpy as np
r_length = 500 
# avg to make plot more visibly clear
training_error_avg = (
    np.convolve(np.array(agent.training_error), np.ones(r_length), mode = 'same')
    / r_length
)
plt.plot(range(len(training_error_avg)), training_error_avg)
plt.show()