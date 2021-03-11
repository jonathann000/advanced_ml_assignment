import argparse
import gym
import importlib.util

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)
reward = []



env = gym.make("FrozenLake-v0")
action_dim = env.action_space.n
state_dim = env.observation_space.n

agent = agentfile.Agent(state_dim, action_dim)

observation = env.reset()
for _ in range(10000): 
    #env.render()
    action = agent.act(observation) # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    agent.observe(observation, reward, done)


    print(reward)
    if done: 
        observation = env.reset() 
env.close()
