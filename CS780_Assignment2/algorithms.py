import RME_file as RME
import numpy as np
import pandas as pd
import random
import gymnasium
import matplotlib.pyplot as plt
import math
from mazeFigure import MazeEnvironment
from plotVandQ import plots
from utilities import *


def MonteCarloControl(env, gamma, maxSteps, noEpisodes, decayConstAlpha, decayConstEpsilon, typeDecay):
    
    total_states = 12
    total_actions = 4
    q = np.zeros((12, 4))
    
    # stores q for all episodes
    episodic_q = np.zeros((noEpisodes, 12, 4))
   
    # Generate random values for the policy matrix
    policy = np.array([[0.25]*total_actions]*total_states) 
    value_function = np.zeros((12, noEpisodes))
    
    for e in range(noEpisodes):
        
        alpha_initial = decayAlpha(e, decayConstAlpha, typeDecay)
        epsilon_initial = decayEpsilon(e, decayConstEpsilon, typeDecay)
        state_occurred_in_episode = []
        start_state, info = env.reset()
        all_exp = generateTrajectory(env, policy, maxSteps)
        # print(all_exp)
        
        visited = np.array([[False]*total_actions]*total_states)
        
        for i, (s, a, s_next, r) in enumerate(all_exp):
            s = int(s)
            a = int(a)
            state_occurred_in_episode.append(s)
            s_next = int(s_next)
            
            if visited[s,a] == True: #first visit
                continue
            else:
                visited[s,a] = True
                
                G = 0
                for j in range(i, len(all_exp)):
                    G += (gamma**(j-i))*all_exp[j][3]
                
                q[s,a] += alpha_initial*(G - q[s,a])
        episodic_q[e] = q
        
        state_occurred_in_episode = np.array(state_occurred_in_episode)
        for s in state_occurred_in_episode:
            a_greedy = np.argmax(q[s,:])

            for a in range(4):
                if a == a_greedy:
                    policy[s,a] = 1 - epsilon_initial + (epsilon_initial/4)
                else:
                    policy[s,a] = (epsilon_initial/4)

        value_function[:,e] = np.max(q, axis=1)
    return policy, episodic_q, value_function, q



def SARSA(env, gamma, decayConstAlpha, decayConstEpsilon, noEpisodes, typeDecay):
    
    
    total_states =12
    total_actions = 4
    
    
    q = np.zeros((total_states, total_actions), dtype = float)
    # stores q for all episodes
    episodic_q = np.zeros((noEpisodes, total_states, total_actions))
    # Generate random values for the policy matrix
    policy = np.array([[0.25]*total_actions]*total_states) 
    value_function = np.zeros((total_states, noEpisodes))
    
    for e in range(noEpisodes):
        
        alpha_initial = decayAlpha(e, decayConstAlpha, typeDecay)
        epsilon_initial = decayEpsilon(e, decayConstEpsilon, typeDecay)
        state_occurred_in_episode = []
        obs, info = env.reset()
        s = int(info['Start State'])
        state_occurred_in_episode.append(s)
        a = actionSelect(s, q, epsilon_initial)
        
        while info['Termination Status'] == False:
            observation, reward, terminated, truncated, info = env.step(a)
            s_next = int(info['Start State'])
            a_next = actionSelect(s_next, q, 0.1)
            
            td_target = reward
            if terminated == False:
                td_target += gamma*q[s_next, a_next]
            
            td_err = 0
            td_err = td_target - q[s, a]
            update = alpha_initial*td_err
            initial = q[s,a]
            q[s, a] = initial + update
            s = s_next
            state_occurred_in_episode.append(s)
            a = a_next
        
        episodic_q[e] = q
        state_occurred_in_episode = np.array(state_occurred_in_episode)
        # print(state_occurred_in_episode)
        for s in state_occurred_in_episode:
            a_greedy = np.argmax(q[s,:])

            for a in range(4):
                if a == a_greedy:
                    policy[s,a] = 1 - epsilon_initial + (epsilon_initial/4)
                else:
                    policy[s,a] = (epsilon_initial/4)

        value_function[:,e] = np.max(q, axis=1)
    return policy, episodic_q, value_function, q     


def Qlearning(env, gamma, decayConstAlpha, decayConstEpsilon, noEpisodes, typeDecay):
    
    total_states =12
    total_actions = 4
    
    
    q = np.zeros((total_states, total_actions), dtype = float)
    # stores q for all episodes
    episodic_q = np.zeros((noEpisodes, total_states, total_actions))
    # Generate random values for the policy matrix
    policy = np.array([[0.25]*total_actions]*total_states) 
    value_function = np.zeros((total_states, noEpisodes))
    
    for e in range(noEpisodes):
        
        alpha_initial = decayAlpha(e, decayConstAlpha, typeDecay)
        epsilon_initial = decayEpsilon(e, decayConstEpsilon, typeDecay)
        obs, info = env.reset()
        state_occurred_in_episode = []
        s = info['Start State']
        state_occurred_in_episode.append(s)
        
        while info['Termination Status'] == False:
            a = actionSelect(s, q, epsilon_initial)
            observation, reward, terminated, truncated, info = env.step(a)
            s_next = info['Start State']
            
            td_target = reward
            if info['Termination Status'] == False:
                td_target += gamma*max(q[s_next])
            
            td_error = td_target - q[s, a]
            q[s, a] += alpha_initial*td_error
            s = s_next
            state_occurred_in_episode.append(s)
            
        
        episodic_q[e] = q
        state_occurred_in_episode = np.array(state_occurred_in_episode)
        for s in state_occurred_in_episode:
            a_greedy = np.argmax(q[s,:])

            for a in range(4):
                if a == a_greedy:
                    policy[s,a] = 1 - epsilon_initial + (epsilon_initial/4)
                else:
                    policy[s,a] = (epsilon_initial/4)

        value_function[:,e] = np.max(q, axis=1)
    return policy, episodic_q, value_function, q


def doubleQlearning(env, gamma, decayConstAlpha, decayConstEpsilon, noEpisodes,typeDecay):
    total_states =12
    total_actions = 4
    
    
    q1 = np.zeros((total_states, total_actions), dtype = float)
    q2 = np.zeros((total_states, total_actions), dtype = float)
    q = np.zeros((total_states, total_actions), dtype = float)
    
    # stores q for all episodes
    episodic_q1 = np.zeros((noEpisodes, total_states, total_actions))
    episodic_q2 = np.zeros((noEpisodes, total_states, total_actions))
    episodic_q = np.zeros((noEpisodes, total_states, total_actions))
    
    # Generate random values for the policy matrix
    policy = np.array([[0.25]*total_actions]*total_states) 
    value_function = np.zeros((total_states, noEpisodes))
    
    for e in range(noEpisodes):
        
        alpha_initial = decayAlpha(e, decayConstAlpha, typeDecay)
        epsilon_initial = decayEpsilon(e, decayConstEpsilon, typeDecay)
        obs, info = env.reset()
        state_occurred_in_episode = []
        s = info['Start State']
        state_occurred_in_episode.append(s)
        
        while info['Termination Status'] == False:
            a = actionSelect(s, q, epsilon_initial)
            observation, reward, terminated, truncated, info = env.step(a)
            s_next = info['Start State']
            
            if random.randint(0, 1):
                a_q1 = np.argmax(q1[s_next])
                td_target = reward
                
                if terminated == False:
                    td_target += gamma*q2[s_next, a_q1]
                
                td_error = td_target - q1[s,a]
                q1[s,a] += alpha_initial*td_error
            else:
                a_q2 = np.argmax(q2[s_next])
                td_target = reward
                
                if terminated == False:
                    td_target += gamma*q1[s_next, a_q2]
                
                td_error = td_target - q2[s,a]
                q2[s,a] += alpha_initial*td_error
            
            s = s_next
            state_occurred_in_episode.append(s)
            
        episodic_q1[e] = q1
        episodic_q2[e] = q2
        
        q = (q1+q2)/2
        episodic_q = (episodic_q1 + episodic_q2)/2
        state_occurred_in_episode = np.array(state_occurred_in_episode)
        
        for s in state_occurred_in_episode:
            a_greedy = np.argmax(q[s,:])

            for a in range(4):
                if a == a_greedy:
                    policy[s,a] = 1 - epsilon_initial + (epsilon_initial/4)
                else:
                    policy[s,a] = (epsilon_initial/4)

        value_function[:,e] = np.max(q, axis=1)
    return policy, episodic_q, value_function, q

def SARSA_Lambda(env, gamma, decayConstAlpha, decayConstEpsilon, lambda_, noEpisodes, replaceTrace, typeDecay):
    total_states =12
    total_actions = 4
    
    
    q = np.zeros((total_states, total_actions), dtype = float)
    exp = np.zeros((total_states, total_actions), dtype = float)
    
    
    # stores q for all episodes
    episodic_q = np.zeros((noEpisodes, total_states, total_actions))
    # Generate random values for the policy matrix
    policy = np.array([[0.25]*total_actions]*total_states) 
    value_function = np.zeros((total_states, noEpisodes))
    
    for e in range(noEpisodes):
        
        alpha_initial = decayAlpha(e, decayConstAlpha, typeDecay)
        epsilon_initial = decayEpsilon(e, decayConstEpsilon, typeDecay)
        state_occurred_in_episode = []
        obs, info = env.reset()
        s = info['Start State']
        state_occurred_in_episode.append(s)
        a = actionSelect(s, q, epsilon_initial)
        exp.fill(0)
        
        while info['Termination Status'] == False:
            observation, reward, terminated, truncated, info = env.step(a)
            s_next = info['Start State']
            
            a_next = actionSelect(s_next, q, epsilon_initial)
            td_target = reward
            
            if terminated ==False:
                td_target += gamma*q[s_next, a_next]
            
            td_error = td_target - q[s,a]
            exp[s, a] += 1
            
            if replaceTrace == True:
                exp[s, a] = 1
            
            q += alpha_initial*td_error*exp
            
            exp = gamma*lambda_*exp
            s = s_next
            a = a_next
            state_occurred_in_episode.append(s)
        
        episodic_q[e] = q
        state_occurred_in_episode = np.array(state_occurred_in_episode)
        
        for s in state_occurred_in_episode:
            a_greedy = np.argmax(q[s,:])

            for a in range(4):
                if a == a_greedy:
                    policy[s,a] = 1 - epsilon_initial + (epsilon_initial/4)
                else:
                    policy[s,a] = (epsilon_initial/4)

        value_function[:,e] = np.max(q, axis=1)
    return policy, episodic_q, value_function, q        



def Q_lambda(env, gamma, decayConstAlpha, decayConstEpsilon, lambda_, noEpisodes, replaceTrace, typeDecay):
    total_states =12
    total_actions = 4
    
    
    q = np.zeros((total_states, total_actions), dtype = float)
    exp = np.zeros((total_states, total_actions), dtype=float)
    
    # stores q for all episodes
    episodic_q = np.zeros((noEpisodes, total_states, total_actions))
    # Generate random values for the policy matrix
    policy = np.array([[0.25]*total_actions]*total_states) 
    value_function = np.zeros((total_states, noEpisodes))
    
    for e in range(noEpisodes):
        
        alpha_initial = decayAlpha(e, decayConstAlpha, typeDecay)
        epsilon_initial = decayEpsilon(e, decayConstEpsilon, typeDecay)
        state_occurred_in_episode = []
        obs, info = env.reset()
        s = int(info['Start State'])
        state_occurred_in_episode.append(s)
        a = actionSelect(s, q, epsilon_initial)
        exp = np.zeros((total_states, total_actions), dtype=float)

        
        while info['Termination Status'] == False:
            observation, reward, terminated, truncated, info = env.step(a)
            s_next = int(info['Start State'])
            
            a_next = actionSelect(s_next, q, epsilon_initial)
            if q[s_next, a_next] == max(q[s_next, :]):
                s_next_greedy = True
            else:
                s_next_greedy = False
            
            
            td_target = reward
            
            if terminated ==False:
                td_target += gamma*max(q[s_next, :])
            
            td_error = td_target - q[s,a]
            
            if replaceTrace:
                exp[s, :] = 0
                
            exp[s, a] += 1
            
            q += alpha_initial*td_error*exp
            
            if s_next_greedy == True:
                exp = gamma*lambda_*exp
            else:
                exp = np.array([[0]*total_actions]*total_states)

            s = s_next
            a = a_next
            state_occurred_in_episode.append(s)
        
        episodic_q[e] = q
        state_occurred_in_episode = np.array(state_occurred_in_episode)
        
        for s in state_occurred_in_episode:
            a_greedy = np.argmax(q[s,:])

            for a in range(4):
                if a == a_greedy:
                    policy[s,a] = 1 - epsilon_initial + (epsilon_initial/4)
                else:
                    policy[s,a] = (epsilon_initial/4)

        value_function[:,e] = np.max(q, axis=1)
    return policy, episodic_q, value_function, q        



def dynaQ(env, gamma, noEpisodes, noPlanning,decayConstAlpha, decayConstEpsilon, typeDecay):
    total_states =12
    total_actions = 4
    
    
    q = np.zeros((total_states, total_actions), dtype = float)    
    # stores q for all episodes
    episodic_q = np.zeros((noEpisodes, total_states, total_actions))
    t = np.zeros((total_states, total_actions, total_states))
    r = np.zeros((total_states, total_actions, total_states))
    # Generate random values for the policy matrix
    policy = np.array([[0.25]*total_actions]*total_states)
    value_function = np.zeros((total_states, noEpisodes))

    
    for e in range(noEpisodes):
        
        alpha_initial = decayAlpha(e, decayConstAlpha, typeDecay)
        epsilon_initial = decayEpsilon(e, decayConstEpsilon, typeDecay)
        #decay alpha and epsilon
        state_occurred_in_episode = []
        obs, info = env.reset()
        s = info['Start State']
        state_occurred_in_episode.append(s)
        
        while info['Termination Status'] == False:
            a = actionSelect(s, q, epsilon_initial)
            observation, reward, terminated, truncated, info = env.step(a)

            s_next = info['Start State']
            
            t_old = t
            t[s, a, s_next] += 1
            rDiff = reward - r[s,a,s_next]
            r[s,a,s_next] += rDiff/t[s,a,s_next]

            td_target = reward
            if info['Termination Status'] == False:
                td_target += gamma*max(q[s_next,:])

            td_error = td_target - q[s,a]
            q[s,a] += alpha_initial*td_error
            s_backup = s_next

            for _ in range(noPlanning):
                if np.all(q == 0):
                    break
                s_visited, a_taken = getVisitedStatesAndActionsTaken(t, t_old)
                if s_visited.size != 0 and a_taken.size !=0:
                    s = random.choice(s_visited)
                    a = random.choice(a_taken)
                    prob_s_next = t[s,a]/sum(T[s,a,:])
                    s_next = np.random.choice(range(12), p=prob_s_next)
                    reward = r[s,a,s_next]
                    td_target = reward + gamma*max(q[s_next,:])
                    td_error = td_target - q[s,a]
                    q[s,a] += alpha_initial*td_error

            s = s_backup
            state_occurred_in_episode.append(s)
    
        episodic_q[e] = q
        state_occurred_in_episode = np.array(state_occurred_in_episode)
        
        for s in state_occurred_in_episode:
            a_greedy = np.argmax(q[s,:])

            for a in range(4):
                if a == a_greedy:
                    policy[s,a] = 1 - epsilon_initial + (epsilon_initial/4)
                else:
                    policy[s,a] = (epsilon_initial/4)

        value_function[:,e] = np.max(q, axis=1)
    return policy, episodic_q, value_function, q




def trajectorySampling(env, gamma, decayConstAlpha, decayConstEpsilon, maxTrajectory, noEpisodes, typeDecay):
    total_states =12
    total_actions = 4
    
    
    q = np.zeros((total_states, total_actions), dtype = float)    
    # stores q for all episodes
    episodic_q = np.zeros((noEpisodes, total_states, total_actions), dtype = float)
    t = np.zeros((total_states, total_actions, total_states), dtype = float)
    r = np.zeros((total_states, total_actions, total_states), dtype = float)
    # Generate random values for the policy matrix
    policy = np.array([[0.25]*total_actions]*total_states)
    value_function = np.zeros((total_states, noEpisodes))
    
    
    for e in range(noEpisodes):
        
        alpha_initial = decayAlpha(e, decayConstAlpha, typeDecay)
        epsilon_initial = decayEpsilon(e, decayConstEpsilon, typeDecay)
        #decay alpha and epsilon
        state_occurred_in_episode = []
        obs, info = env.reset()
        s = info['Start State']
        state_occurred_in_episode.append(s)
        
        while info['Termination Status'] == False:
            a = actionSelect(s, q, epsilon_initial)
            observation, reward, terminated, truncated, info = env.step(a)

            s_next = info['Start State']
            
            t_old = t
            t[s, a, s_next] += 1
            rDiff = reward - r[s,a,s_next]
            r[s,a,s_next] += rDiff/t[s,a,s_next]

            td_target = reward
            if info['Termination Status'] == False:
                td_target += gamma*max(q[s_next,:])

            td_error = td_target - q[s,a]
            q[s,a] += alpha_initial*td_error
            s_backup = s_next

            for _ in range(maxTrajectory):
                if np.all(q == 0):
                    break
                a = np.argmax(q[s,:])
                
                if np.all(t[s,a]  == 0):
                    break
                prob_s_next = t[s,a]/sum(t[s,a,:])
                s_next = np.random.choice(range(12), p=prob_s_next)
                reward = r[s,a,s_next]
                td_target = reward + gamma*max(q[s_next,:])
                td_error = td_target - q[s,a]
                q[s,a] += alpha_initial*td_error
                s = s_next
            s = s_backup
            state_occurred_in_episode.append(s)
    
        episodic_q[e] = q
        state_occurred_in_episode = np.array(state_occurred_in_episode)
        
        for s in state_occurred_in_episode:
            a_greedy = np.argmax(q[s,:])

            for a in range(4):
                if a == a_greedy:
                    policy[s,a] = 1 - epsilon_initial + (epsilon_initial/4)
                else:
                    policy[s,a] = (epsilon_initial/4)

        value_function[:,e] = np.max(q, axis=1)
    return policy, episodic_q, value_function, q