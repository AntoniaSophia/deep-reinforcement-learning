import logging
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


LOG_FMT = logging.Formatter('%(levelname)s '
                            '[%(filename)s:%(lineno)d] %(message)s',
                            '%Y-%m-%d %H:%M:%S')

N_DISCRETE_ACTIONS = 6
NUM_WHEEL_ElEMENTS = 10
NUM_OF_TURNS = 5
N_DISCRETE_STATES = NUM_WHEEL_ElEMENTS * NUM_OF_TURNS
#PROP_TURN = [[0.3 , 0.4 , 0.3] , [0.25 , 0.5 , 0.25], [0.15 , 0.7 , 0.15] , [0.1 , 0.8 , 0.1] , [0.05,0.9,0.05], [0,1,0]]
#PROP_TURN = [[0.1 , 0.8 , 0.1] , [0.1 , 0.8 , 0.1],[0.1 , 0.8 , 0.1],[0.1 , 0.8 , 0.1],[0.1 , 0.8 , 0.1], [0,1,0]]
PROP_TURN = [[0,0,1,0,0] , [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0], [0,0,1,0,0]]




class WheelOfFortuneEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}



    def __init__(self):
        super(WheelOfFortuneEnv, self).__init__()    # Define action and observation space
        # They must be gym.spaces objects    # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)    # Example for using image as input:
        self.observation_space = spaces.Tuple((
            spaces.Discrete(NUM_OF_TURNS),
            spaces.Discrete(NUM_WHEEL_ElEMENTS)))
        
        #self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        # print('State shape: ', self.observation_space.shape)
        # print('Number of actions: ', self.action_space.n)

        self.seed()

        self.currentState = (1,1)   # (Turn , Element)
        self.currentAction =  -1
        self.currentIntendedAction = -1
        self.stateHistory = []
        self.stateHistory.append(self.currentState)
        self.reward = 0
        self.done = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def take_action(self, action):
        #action_range = [action-1, action, action +1]
        #prop_action = np.random.choice(action_range, 1, PROP_TURN[self.currentState[0]])

        #if prop_action[0] < 0:
        #    prop_action[0] = 0

        #self.currentAction = prop_action[0]
        
        self.currentAction = action
        self.currentIntendedAction = action


        new_element = self.currentState[1] + self.currentAction 
        
        new_turn = 0

        if new_element > NUM_WHEEL_ElEMENTS:
            new_turn = 1
            new_element = new_element % NUM_WHEEL_ElEMENTS
    
        self.currentState= (self.currentState[0] + new_turn , new_element)

        self.stateHistory.append(self.currentState)


    def step(self, action):
        if self.done:
            return self.currentState, 0, self.done, {}



        reward = -5

        # Execute one time step within the environment
        #self.take_action(action)
        self.take_action(action)

        if self.currentIntendedAction == 0:
            reward = -20

        if self.currentState[0] == 2 and self.currentState[1] % 2 == 0:
            reward += 1
        if self.currentState[0] == 3 and self.currentState[1] % 2 == 1:
            reward += 1
        if self.currentState[0] == 4 and self.currentState[1] % 2 == 0:
            reward += 1
        if self.currentState[0] == 5 and self.currentState[1] % 2 == 1:
            reward += 1

        #obs = self._next_observation()
        obs = self.currentState

        # calculate done
        if self.currentState[0] > NUM_OF_TURNS:
            self.done = True 

        # calculate reward
        if self.done:
            min_turn_2 = 1000
            min_turn_3 = 1000
            min_turn_4 = 1000
            min_turn_5 = 1000

            for turn,element in self.stateHistory:
                if turn == 2 and element < min_turn_2:
                    min_turn_2 = element
                if turn == 3 and element < min_turn_3:
                    min_turn_3 = element
                if turn == 4 and element < min_turn_4:
                    min_turn_4 = element
                if turn == 5 and element < min_turn_5:
                    min_turn_5 = element

            # print("min_turn_2 = " , min_turn_2)
            # print("min_turn_3 = " , min_turn_3)
            # print("min_turn_4 = " , min_turn_4)
            # print("min_turn_5 = " , min_turn_5)

            #if min_turn_2 == min_turn_3 and min_turn_3 == min_turn_4 and min_turn_4 == min_turn_5:
            #    reward = 100

            # Another attemp for reward function - pretty easy...
            #reward += min_turn_2 * 1 + min_turn_3 * 3 + min_turn_4 * 5 + min_turn_5 * 10

            # Another attemp for reward function - pretty easy...
            #reward += min_turn_2 * (-1) + min_turn_3 * (-1) + min_turn_4 * (-1) + min_turn_5 * 100


            for turn,element in self.stateHistory:
                if turn == 2 and element % 2 == 0:
                    reward += 15
                if turn == 3 and element % 2 == 1:
                    reward += 15
                if turn == 4 and element % 2 == 0:
                    reward += 15
                if turn == 5 and element % 2 == 1:
                    reward += 15
           
        self.reward += reward
        
        return obs, reward, self.done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.currentState = (1,1)
        self.stateHistory.clear()
        self.stateHistory.append(self.currentState)
        self.currentAction =  -1
        self.done = False
        self.reward = 0
        self.currentIntendedAction = -1
        
        return self.currentState
        
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'---------------------------------------------')
        print(f'Current Intended Action: {self.currentIntendedAction}')
        print(f'Current Action: {self.currentAction}')
        print(f'Next State: {self.currentState}')
        print(f'Sum Reward: {self.reward}')
        print(f'State History: {self.stateHistory}')
        print(f'Done: {self.done}')



def set_log_level_by(verbosity):
    """Set log level by verbosity level.
    verbosity vs log level:
        0 -> logging.ERROR
        1 -> logging.WARNING
        2 -> logging.INFO
        3 -> logging.DEBUG
    Args:
        verbosity (int): Verbosity level given by CLI option.
    Returns:
        (int): Matching log level.
    """
    if verbosity == 0:
        level = 40
    elif verbosity == 1:
        level = 30
    elif verbosity == 2:
        level = 20
    elif verbosity >= 3:
        level = 10

    logger = logging.getLogger()
    logger.setLevel(level)
    if len(logger.handlers):
        handler = logger.handlers[0]
    else:
        handler = logging.StreamHandler()
        logger.addHandler(handler)

    handler.setLevel(level)
    handler.setFormatter(LOG_FMT)
    return level


# if __name__ == '__main__':
#     env = WheelOfFortuneEnv()

#     env.render()
#     env.step(3)
#     env.render()
#     env.step(1)
#     env.render()
#     env.step(2)
#     env.render()
#     env.step(3)
#     env.render()
#     env.step(2)
#     env.render()
#     env.step(1)
#     env.render()
#     env.step(2)
#     env.render()
#     env.step(3)
#     env.render()
#     env.step(1)
#     env.render()
#     env.step(2)
#     env.render()
#     env.step(3)
#     env.render()
#     env.step(1)
#     env.render()
#     env.step(3)
#     env.render()