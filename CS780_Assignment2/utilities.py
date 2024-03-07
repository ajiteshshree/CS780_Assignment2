import numpy as np
import pandas as pd
import math
import random



#changes stochastic policy to deterministic policy
def deterministicPolicy(stochasticPolicy):
    action_ = []
    deterministicPolicy = []
    for state in range(12):
        determinisntic = max(stochasticPolicy[state])
        action = np.argmax(stochasticPolicy[state])
        deterministicPolicy.append(determinisntic)
        action_.append(action)
    
    return np.array(deterministicPolicy), np.array(action_)



def decayAlpha(episode_number, decayConst, type):
    if type == 'exponential':
        alpha = math.exp(-decayConst*episode_number)
    elif type == 'linear':
        alpha = 1 - decayConst*episode_number
    return alpha



def decayEpsilon(episode_number, decayConst, type):
    if type == 'exponential':
        epsilon = math.exp(-decayConst*episode_number)
    elif type == 'linear':
        epsilon = 1 - decayConst*episode_number
    return epsilon



# as epsilon decreases, exploits
def epsilonGreedypolicy(policy_state, epsilon):
    
    number = random.random()
    
    if number < epsilon:  #explore
        action = np.random.randint(4)
    else: #exploit
        action = np.argmax(policy_state)
    
    return action


def generateTrajectory(env, policy, maxSteps):
    
    start_state, info = env.reset()
    episode = 0
    completed_episode = 0
    all_experience = [] 
    initial_state = info['Start State']
    total_reward = info['Reward']
    
    for i in range(maxSteps):
        
        state = info['Start State']
        
        action = epsilonGreedypolicy(policy_state = policy[state], epsilon = 0.1)
        observation, reward, terminated, truncated, info = env.step(action)
        
        experience = observation
        all_experience.append(experience)
        
        if terminated:
            start_state, info = env.reset()
            return np.array(all_experience)
        
        if i == maxSteps - 1 and not terminated:
            # truncated = True, so run it again till it gets terminated,
            # i.e. one complete episode completed always 
            generateTrajectory(env, policy, maxSteps)
    
def actionSelect(state, q, epsilon):
    number = random.random()
    
    #for small epsilon, more exploit
    if number < epsilon:
        action = random.randint(0,3) #explore
    else:    
        action = np.argmax(q[state]) #exploit
    
    return action

def getVisitedStatesAndActionsTaken(t, t_old):
    s_visited = []
    a_taken = []
    
    for s in range(12):
        for a in range(4):
            for s_next in range(12):
                if t[s,a, s_next] != t_old[s,a,s_next]:
                    s_visited.append(s_next)
                    a_taken.append(a)
    
    s_visited = np.array(s_visited)
    a_taken = np.array(a_taken)
    
    return s_visited, a_taken