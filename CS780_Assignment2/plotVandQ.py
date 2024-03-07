import matplotlib.pyplot as plt
import numpy as np
from utilities import *
from mazeFigure import *
import pandas as pd


def plots(value_function, episodic_q, noEpisodes):

    for i in range(12):
        if i != 5 and i != 7 and i != 3:
            plt.plot(value_function[i, range(noEpisodes)], label = f'State {i}')
    plt.xlabel('Episodes')        
    plt.ylabel('Value of state')
    plt.title(f'Value of state throughout {noEpisodes} episodes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    for i in range(12):
        if i != 5 and i != 7 and i!=3:
            plt.scatter(i, value_function[i, noEpisodes -1], label=f'State {i}: {value_function[i, noEpisodes-1]:.2f}')
            plt.vlines(i, ymin=0, ymax=value_function[i, noEpisodes-1], linestyles='dotted', colors='gray')

    plt.xticks(np.arange(12))
    plt.xlabel('States')        
    plt.ylabel('True Value of states')
    plt.title(f'True value of state, i.e. value at last episode')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs = axs.flatten()
    subplot_index = 0

    for state in range(12):
        if state not in [3, 5, 7]:
            for a in range(4):
                axs[subplot_index].plot(range(noEpisodes), episodic_q[range(noEpisodes), state, a], label=f'Action {a}')


            axs[subplot_index].set_xlabel('Episodes')
            axs[subplot_index].set_ylabel(f'State-Action value for state {state}')
            axs[subplot_index].set_title(f'State-Action value throughout {noEpisodes} episodes')

            subplot_index += 1


    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='lower left')
    plt.show()


    colors = plt.cm.viridis(np.linspace(0, 1, 4))  # Adjust the colormap as needed
    break_ = 0
    for state in range(12):
        if state not in [3, 5, 7]:
            for a in range(4):
                plt.scatter(state, episodic_q[noEpisodes - 1, state, a], label=f'True Q-value for action {a}', color=colors[a])
                plt.vlines(state, ymin= min(episodic_q[noEpisodes - 1, state, :]), ymax=episodic_q[noEpisodes - 1, state, a], linestyles='dotted', colors='gray')
            if break_ == 0:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='center left')
                break_ = 1

    plt.xticks(np.arange(12))
    plt.xlabel('State')
    plt.ylabel('True Q-value')
    plt.title('True Q-value for each state at last episode')
    plt.show()
    
    
    
    
def showPlot(policy, value_function, episodic_q, noEpisodes):
    maze = MazeEnvironment()
    detP, corrA = deterministicPolicy(policy)
    print(f'Stochastic Policy:\n{pd.DataFrame(policy)}\n')
    print(f'Deterministic Policy:\n{detP}\n')

    print('Actions according to deterministic policy:', corrA)
    maze.display_maze(corrA)
    print('\n------------------------------------------------------------------------------------------')
    plots(value_function,episodic_q, noEpisodes)