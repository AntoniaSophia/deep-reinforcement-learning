from gym.envs.registration import register
 
register(id='wheel-v0', 
    entry_point='gym_wheel.envs:WheelOfFortuneEnv', 
)