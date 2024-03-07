from gymnasium import Env, spaces, register, make
import numpy as np
import random

class CustomMultiDiscrete(spaces.MultiDiscrete):
    def __init__(self, nvec):
        super().__init__(nvec)

    def sample(self):
        sample = super().sample()
        sample = np.float64(sample)

        # Modify the last value
        if sample[-1] == 3:
            sample[-1] = -1
        elif sample[-1] == 2:
            sample[-1] = -0.04
        
            
        return sample

class RandomMazeEnvironment(Env):
    def __init__(self):
        
        # at terminal state, whatever action taken leads to 0 reward and no state transition with  1 probability
        # no need to define wall state 5 as agent will never be there
        # if action is hitting two boundaries with 0.8, 0.1 percentage occuring, 0.9 precentage chance defined to be in the current state itself.
        
        self.possibilities = {
            0: { #[(transition_prob, next state, reward, termination_status)]
                0: [(0.9, 0, -0.04, False), (0.1, 1, -0.04, False)],
                1: [(0.8, 1, -0.04, False),(0.1, 4, -0.04, False),(0.1, 0, -0.04, False)],
                2: [(0.8, 4, -0.04, False),(0.1, 1, -0.04, False),(0.1, 0, -0.04, False)],
                3: [(0.9, 0, -0.04, False), (0.1, 4, -0.04, False)]
            },
            1: {
                0: [(0.8, 1, -0.04, False), (0.1, 2, -0.04, False), (0.1, 0, -0.04, False)],
                1: [(0.8, 2, -0.04, False),(0.2, 1, -0.04, False)],
                2: [(0.8, 1, -0.04, False), (0.1, 2, -0.04, False), (0.1, 0, -0.04, False)],
                3: [(0.8, 0, -0.04, False),(0.2, 1, -0.04, False)]
            },
            2: {
                0: [(0.8, 2, -0.04, False), (0.1, 3, 1.0, True), (0.1, 1, -0.04, False)],
                1: [(0.8, 3, 1.0, True),(0.1, 6, -0.04, False),(0.1, 2, -0.04, False)],
                2: [(0.8, 6, -0.04, False),(0.1, 3, 1.0, True),(0.1, 1, -0.04, False)],
                3: [(0.8, 1, -0.04, False),(0.1, 6, -0.04, False), (0.1, 2, -0.04, False)]
            },
            3: {
                0: [(1.0, 3, 0, True)],
                1: [(1.0, 3, 0, True)],
                2: [(1.0, 3, 0, True)],
                3: [(1.0, 3, 0, True)]
            },
            4: {
                0: [(0.8, 0, -0.04, False),(0.2, 4, -0.04, False)],
                1: [(0.8, 4, -0.04, False),(0.1, 8, -0.04, False),(0.1, 0, -0.04, False)],
                2: [(0.8, 8, -0.04, False),(0.2, 4, -0.04, False)],
                3: [(0.8, 4, -0.04, False),(0.1, 0, -0.04, False), (0.1, 8, -0.04, False)]
            },
            6: {
                0: [(0.8, 2, -0.04, False),(0.1, 7, -1.0, True), (0.1, 6, -0.04, False)],
                1: [(0.8, 7, -1.0, True),(0.1, 10, -0.04, False),(0.1, 2, -0.04, False)],
                2: [(0.8, 10, -0.04, False),(0.1, 7, -1.0, True), (0.1, 6, -0.04, False)],
                3: [(0.8, 6, -0.04, False),(0.1, 10, -0.04, False), (0.1, 2, -0.04, False)]
            },
            7: {
                0: [(1.0, 7, 0, True)],
                1: [(1.0, 7, 0, True)],
                2: [(1.0, 7, 0, True)],
                3: [(1.0, 7, 0, True)]
            },
            8: {
                0: [(0.8, 4, -0.04, False),(0.1, 9, -0.04, False),(0.1, 8, -0.04, False)],
                1: [(0.8, 9, -0.04, False),(0.1, 8, -0.04, False),(0.1, 4, -0.04, False)],
                2: [(0.9, 8, -0.04, False),(0.1, 9, -0.04, False)],
                3: [(0.9, 8, -0.04, False), (0.1, 4, -0.04, False)]
            },
            9: {
                0: [(0.8, 9, -0.04, False),(0.1, 8, -0.04, False),(0.1, 10, -0.04, False)],
                1: [(0.8, 10, -0.04, False),(0.2, 9, -0.04, False)],
                2: [(0.8, 9, -0.04, False),(0.1, 10, -0.04, False),(0.1, 8, -0.04, False)],
                3: [(0.8, 8, -0.04, False),(0.2, 9, -0.04, False)]
            },
            10: {
                0: [(0.8, 6, -0.04, False),(0.1, 9, -0.04, False),(0.1, 11, -0.04, False)],
                1: [(0.8, 11, -0.04, False),(0.1, 6, -0.04, False),(0.1, 10, -0.04, False)],
                2: [(0.8, 10, -0.04, False),(0.1, 9, -0.04, False),(0.1, 11, -0.04, False)],
                3: [(0.8, 9, -0.04, False),(0.1, 6, -0.04, False), (0.1, 10, -0.04, False)]
            },
            11: {
                0: [(0.8, 7, -1.0, True),(0.1, 11, -0.04, False),(0.1, 10, -0.04, False)],
                1: [(0.9, 11, -0.04, False),(0.1, 7, -1.0, True)],
                2: [(0.9, 11, -0.04, False),(0.1, 10, -0.04, False)],
                3: [(0.8, 10, -0.04, False),(0.1, 7, -1.0, True), (0.1, 11, -0.04, False)]
            }
            
        }
        # observe: [state, action, nextState, reward]
        # self.observation_space = spaces.MultiDiscrete([12, 4, 12, 4])
        self.observation_space = CustomMultiDiscrete([12, 4, 12, 4])    
            
        # up - 0, right - 1, down - 2, left - 3
        self.action_space = spaces.Discrete(4)
        
    def reset(self, seed= None, options= None):
        super().reset(seed = seed)
        self._agent_location = 8
        self._target_location = 3
        self._dead_state = 7
        self.wall = 5

        prev_location = self._agent_location
        # action = spaces.Discrete(4).sample() #reset using random action instead of a fixed action. 
        # transitions = self.possibilities[prev_location][action]
        # probabilities, next_states, rewards, terminals = zip(*transitions)

        # Randomly select a transition based on the probabilities of transition
        # index = random.choices(range(len(probabilities)), weights=probabilities, k=1)[0]
        
        # self._agent_location, reward, terminated = next_states[index], rewards[index], terminals[index]
        # observation = np.array([8, action, self._agent_location, reward])
        
        # 'Start State': 8, 
        # 'Action': action, 
        # 'Next State': self._agent_location, 
        # 'Reward': reward, 
        # 'Termination Status': terminated
        
        info = {'Start State': self._agent_location, 'Reward': 0, 'Termination Status': False}

        return self._agent_location, info
    
    def step(self, action):

        prev_location = self._agent_location
        transitions = self.possibilities[prev_location][action]
        probabilities, next_states, rewards, terminals = zip(*transitions)

        # Randomly select a transition based on the transition probabilities
        index = random.choices(range(len(probabilities)), weights=probabilities, k=1)[0]
        self._agent_location, reward, terminated = next_states[index], rewards[index], terminals[index]

        truncated = False #if max time-step reached but episode has not ended (agent not in terminal state), then episode is truncated.
        
        observation = np.array([prev_location, action, self._agent_location, reward])
        info = {'Start State': prev_location, 
                'Action': action, 
                'Next State': self._agent_location, 
                'Reward': reward, 
                'Termination Status': terminated}

        return observation, reward, terminated, truncated, info 

    
Possibilities = {
            0: { #[(transition_prob, next state, reward, termination_status)]
                0: [(0.9, 0, -0.04, False), (0.1, 1, -0.04, False)],
                1: [(0.8, 1, -0.04, False),(0.1, 4, -0.04, False),(0.1, 0, -0.04, False)],
                2: [(0.8, 4, -0.04, False),(0.1, 1, -0.04, False),(0.1, 0, -0.04, False)],
                3: [(0.9, 0, -0.04, False), (0.1, 4, -0.04, False)]
            },
            1: {
                0: [(0.8, 1, -0.04, False), (0.1, 2, -0.04, False), (0.1, 0, -0.04, False)],
                1: [(0.8, 2, -0.04, False),(0.2, 1, -0.04, False)],
                2: [(0.8, 1, -0.04, False), (0.1, 2, -0.04, False), (0.1, 0, -0.04, False)],
                3: [(0.8, 0, -0.04, False),(0.2, 1, -0.04, False)]
            },
            2: {
                0: [(0.8, 2, -0.04, False), (0.1, 3, 1.0, True), (0.1, 1, -0.04, False)],
                1: [(0.8, 3, 1.0, True),(0.1, 6, -0.04, False),(0.1, 2, -0.04, False)],
                2: [(0.8, 6, -0.04, False),(0.1, 3, 1.0, True),(0.1, 1, -0.04, False)],
                3: [(0.8, 1, -0.04, False),(0.1, 6, -0.04, False), (0.1, 2, -0.04, False)]
            },
            3: {
                0: [(1.0, 3, 0, True)],
                1: [(1.0, 3, 0, True)],
                2: [(1.0, 3, 0, True)],
                3: [(1.0, 3, 0, True)]
            },
            4: {
                0: [(0.8, 0, -0.04, False),(0.2, 4, -0.04, False)],
                1: [(0.8, 4, -0.04, False),(0.1, 8, -0.04, False),(0.1, 0, -0.04, False)],
                2: [(0.8, 8, -0.04, False),(0.2, 4, -0.04, False)],
                3: [(0.8, 4, -0.04, False),(0.1, 0, -0.04, False), (0.1, 8, -0.04, False)]
            },
            6: {
                0: [(0.8, 2, -0.04, False),(0.1, 7, -1.0, True), (0.1, 6, -0.04, False)],
                1: [(0.8, 7, -1.0, True),(0.1, 10, -0.04, False),(0.1, 2, -0.04, False)],
                2: [(0.8, 10, -0.04, False),(0.1, 7, -1.0, True), (0.1, 6, -0.04, False)],
                3: [(0.8, 6, -0.04, False),(0.1, 10, -0.04, False), (0.1, 2, -0.04, False)]
            },
            7: {
                0: [(1.0, 7, 0, True)],
                1: [(1.0, 7, 0, True)],
                2: [(1.0, 7, 0, True)],
                3: [(1.0, 7, 0, True)]
            },
            8: {
                0: [(0.8, 4, -0.04, False),(0.1, 9, -0.04, False),(0.1, 8, -0.04, False)],
                1: [(0.8, 9, -0.04, False),(0.1, 8, -0.04, False),(0.1, 4, -0.04, False)],
                2: [(0.9, 8, -0.04, False),(0.1, 9, -0.04, False)],
                3: [(0.9, 8, -0.04, False), (0.1, 4, -0.04, False)]
            },
            9: {
                0: [(0.8, 9, -0.04, False),(0.1, 8, -0.04, False),(0.1, 10, -0.04, False)],
                1: [(0.8, 10, -0.04, False),(0.2, 9, -0.04, False)],
                2: [(0.8, 9, -0.04, False),(0.1, 10, -0.04, False),(0.1, 8, -0.04, False)],
                3: [(0.8, 8, -0.04, False),(0.2, 9, -0.04, False)]
            },
            10: {
                0: [(0.8, 6, -0.04, False),(0.1, 9, -0.04, False),(0.1, 11, -0.04, False)],
                1: [(0.8, 11, -0.04, False),(0.1, 6, -0.04, False),(0.1, 10, -0.04, False)],
                2: [(0.8, 10, -0.04, False),(0.1, 9, -0.04, False),(0.1, 11, -0.04, False)],
                3: [(0.8, 9, -0.04, False),(0.1, 6, -0.04, False), (0.1, 10, -0.04, False)]
            },
            11: {
                0: [(0.8, 7, -1.0, True),(0.1, 11, -0.04, False),(0.1, 10, -0.04, False)],
                1: [(0.9, 11, -0.04, False),(0.1, 7, -1.0, True)],
                2: [(0.9, 11, -0.04, False),(0.1, 10, -0.04, False)],
                3: [(0.8, 10, -0.04, False),(0.1, 7, -1.0, True), (0.1, 11, -0.04, False)]
            }
            
        }