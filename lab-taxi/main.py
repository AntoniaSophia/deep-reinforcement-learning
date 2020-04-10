from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
agent = Agent(env.nA)
avg_rewards, best_avg_reward = interact(env, agent)
#print("Mode = " , agent.mode)
#print("Epsilon = " , agent.epsilon)
#print("Gamma = " , agent.gamma)
#print("Alpha = " , agent.alpha)

