import copy
import random
from matplotlib import pyplot as plt

grid_size = [4, 12]

start_point = [3, 0]

final_point = [3, 11]

gamma = 1

alpha = 0.05

actions = [[-1, 0], #north
           [0, +1], #east
           [+1, 0], #south
           [0, -1]] #west

arrows = ['↑', '→', '↓', '←']

def is_cliff(state):
    x, y = state
    if x == 3 and y >=1 and y <= 10:
        return True
    return False

def get_next_state_and_reward(state, action):
    x, y = state
    dx, dy = action
    new_x = x + dx
    new_y = y + dy
    x = x if new_x < 0 or new_x >= grid_size[0] else new_x
    y = y if new_y < 0 or new_y >= grid_size[1] else new_y
    
    rstate = [x, y]
    if is_cliff(rstate):
        return start_point, -100
    return rstate, -1

def is_terminal(state):
    if state[0] == final_point[0] and state[1] == final_point[1]:
        return True
    return False

def generate_action_by_epsilon_greedy(state_action_value, state, epsilon=0.5, use_epsilon=True):
    x, y = state 
    len_actions = len(actions)
    max_value = max(state_action_value[x][y])
    max_values_idx = []

    for i in range(len_actions):
        if state_action_value[x][y][i] == max_value:
            max_values_idx.append(i)

    policy = [0 for i in range(len_actions)]

    if use_epsilon:
        for i in range(len_actions):
            if i in max_values_idx:
                policy[i] = (1-epsilon) / len(max_values_idx) + epsilon / len_actions
            else:
                policy[i] = epsilon / len_actions

    else:
        for i in range(len_actions):
            if i in max_values_idx:
                policy[i] = 1 / len(max_values_idx)

    def get_action_by_policy(policy=[0.25, 0.25, 0.25, 0.25]):
        rdn = random.random()
        aggre = 0
        for i in range(len(policy)):
            aggre += policy[i]
            if rdn < aggre:
                return actions[i]

    return get_action_by_policy(policy)


def get_one_episode(state_action_value, on_policy=True):
    state = start_point
    action = generate_action_by_epsilon_greedy(state_action_value, state)
    aggre_reward = 0
    while not is_terminal(state):
        next_state, reward = get_next_state_and_reward(state, action)
        next_action = generate_action_by_epsilon_greedy(state_action_value, next_state, use_epsilon=on_policy)

        x, y = state
        next_x, next_y = next_state
        action_idx = actions.index(action)
        next_action_idx = actions.index(next_action)

        # update
        state_action_value[x][y][action_idx] += alpha * (reward + gamma * state_action_value[next_x][next_y][next_action_idx] - state_action_value[x][y][action_idx])

        state = next_state
        if on_policy:
            action = next_action
        else:
            action = generate_action_by_epsilon_greedy(state_action_value, state)
        
        aggre_reward += reward
    
    return aggre_reward
        

def SARSA(iter_num=500):
    state_action_value = [[[0 for i in range(len(actions))] for i in range(grid_size[1])] for i in range(grid_size[0])]

    ite = 0
    while ite < iter_num:
        get_one_episode(state_action_value)
        ite += 1
    
    print_states_arrow(state_action_value)

def Q_learning(iter_num=500):
    state_action_value = [[[0 for i in range(len(actions))] for i in range(grid_size[1])] for i in range(grid_size[0])]

    ite = 0
    while ite < iter_num:
        get_one_episode(state_action_value, on_policy=False)
        ite += 1
    
    print_states_arrow(state_action_value)


def print_states_arrow(state_action_value):
    print("State Optimal Action:")
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            max_action_value = max(state_action_value[x][y])
            max_action_idx = state_action_value[x][y].index(max_action_value)
            if is_cliff([x,y]):
                print('*', end='   ')
            else:
                print(arrows[max_action_idx], end='   ')
        print('\n')

length = 1000
# SARSA(length)  
Q_learning(length)
