import gym
import gym_wheel

env = gym.make("wheel-v0")

env.reset()
exit()

counter=0
while True:
    counter+=1
    reward=0
    done = False
    env.reset()
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()
    if reward >0:
        break
print(counter)