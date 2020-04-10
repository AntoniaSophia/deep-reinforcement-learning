import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.mode = 1   # 0 = Sarsa , 1 = Sarsamax , 2 = Expected Sarsa

        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.episode = 1

        self.epsilon = 1
        self.gamma = 1
        self.alpha = 0.02
        
        if self.mode == 0:
            pass
        elif self.mode == 1:
            pass
        elif self.mode == 2:
            self.epsilon = 0.0006
        else:
            print("Error: mode {} is not defined!!!!" , self.mode) 

    def get_policy(self, Q_s):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - self.epsilon + (self.epsilon / self.nA)
        return policy_s

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(self.nA, p=self.get_policy(self.Q[state]))

    def step(self, last_state, last_action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if done:
            self.episode += 1

        if self.mode == 0:
            self.epsilon = 1/self.episode
            self.epsilon = max(self.epsilon,0.0001)
            self.step_sarsa(last_state, last_action, reward, next_state, done)
        elif self.mode == 1:
            self.epsilon = 1/self.episode
            #self.epsilon = 0.005
            self.epsilon = max(self.epsilon,0.0001)
            self.step_sarsamax(last_state, last_action, reward, next_state, done)
        elif self.mode == 2:
            self.step_expectedsarsa(last_state, last_action, reward, next_state, done)
    

    def step_sarsa(self, last_state, last_action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        Q_default = 0   # q_default is the default value in case there is no next_state
        next_action = self.select_action(next_state)
    
        if not done:
            self.Q[last_state][last_action] = self.Q[last_state][last_action] + \
                                              self.alpha*(reward + self.gamma*self.Q[next_state][next_action] - \
                                              self.Q[last_state][last_action])
        else:
            self.Q[last_state][last_action] = self.Q[last_state][last_action] + \
                                              self.alpha*(reward + self.gamma*Q_default - \
                                              self.Q[last_state][last_action])



    def step_sarsamax(self, last_state, last_action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        self.Q[last_state][last_action] = self.Q[last_state][last_action] + \
                                            self.alpha*(reward + self.gamma*np.max(self.Q[next_state]) - \
                                            self.Q[last_state][last_action])

    def step_expectedsarsa(self, last_state, last_action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        Q_default = 0   # q_default is the default value in case there is no next_state
        
        policy_s = np.ones(self.nA) * self.epsilon / self.nA  # current policy (for next state S')
        policy_s[np.argmax(self.Q[next_state])] = 1 - self.epsilon + (self.epsilon / self.nA) # greedy action
        # NOTE: np.dot(Q[next_state], policy_s) = scalar product of two vectors Q[next_state] and policy_s which
        #                                         is sum(Q[next_state][1]*policy_s[1] + Q[next_state][2]*policy_s[2] + ...)  
        
        if not done:
            self.Q[last_state][last_action] = self.Q[last_state][last_action] +  \
                                              self.alpha*(reward + self.gamma*np.dot(self.Q[next_state], policy_s) - \
                                              self.Q[last_state][last_action])
        else:
            self.Q[last_state][last_action] = self.Q[last_state][last_action] + \
                                              self.alpha*(reward + self.gamma*Q_default - \
                                              self.Q[last_state][last_action])
