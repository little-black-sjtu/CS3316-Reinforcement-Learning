import copy
import random

grid_size = [6, 6]

terminal = [[0, 1],[5, 5]]

actions = [[-1, 0], #north
           [0, +1], #east
           [+1, 0], #south
           [0, -1]] #west

lam = 1

direction_names = ['↑', '→', '↓', '←']

def get_next_state(state, action):
    x, y = state
    dx, dy = action
    new_x = x + dx
    new_y = y + dy
    x = x if new_x < 0 or new_x >= grid_size[0] else new_x
    y = y if new_y < 0 or new_y >= grid_size[1] else new_y
    return x, y

def is_terminal(state):
    for t in terminal:
        if state[0] == t[0] and state[1] == t[1]:
            return True
    return False

def get_action_by_policy(policy=[0.25, 0.25, 0.25, 0.25]):
    rdn = random.random()
    if rdn < policy[0]:
        return actions[0]
    elif rdn < policy[0] + policy[1]:
        return actions[1]
    elif rdn < policy[0] + policy[1] + policy[2]:
        return actions[2]
    else:
        return actions[3]

def get_random_start_state():
    x = random.randint(0, grid_size[0]-1)
    y = random.randint(0, grid_size[1]-1)
    return x, y

def get_one_episode(states_value, count_appeared_state_num, first_visit=True):
    state = get_random_start_state()
    states_passingby = []
    # 随机遍历得到一个episode
    while not is_terminal(state):
        states_passingby.append(state)
        action = get_action_by_policy()
        state = get_next_state(state, action)
    # 更新状态值函数
    summary = 0
    max_error = 0
    while len(states_passingby) != 0:
        state = states_passingby.pop()
        summary += (-1) * lam
        if state in states_passingby and first_visit:
            continue
        x, y = state
        count_appeared_state_num[x][y] += 1
        error = (summary - states_value[x][y]) / count_appeared_state_num[x][y]
        states_value[x][y] += error

        error = abs(error)
        max_error = error if error > max_error else max_error
    return max_error

def MONTECARLO(converge_num=10000, first_visit=True):
    states_value = [[0 for i in range(grid_size[1])] for i in range(grid_size[0])]
    count_appeared_state_num = [[0 for i in range(grid_size[1])] for i in range(grid_size[0])]
    
    episode_count = 0
    while episode_count < converge_num:
        
        get_one_episode(states_value, count_appeared_state_num, first_visit=first_visit)       

        episode_count += 1
        if episode_count % 1000 == 0:
            print(episode_count)
            print_states_num(states_value)
    
    print_states_num(count_appeared_state_num)
    return states_value

def TD(converge_num=1000000):
    states_value = [[0 for i in range(grid_size[1])] for i in range(grid_size[0])]
    alpha = 0.005
    episode_count = 0
    while episode_count < converge_num:
        start_state = get_random_start_state()
        if is_terminal(start_state):
            continue
        action = get_action_by_policy()
        next_state = get_next_state(start_state, action)
        x, y = start_state
        newx, newy = next_state
        error = alpha * (-1 + lam * states_value[newx][newy] - states_value[x][y])
        states_value[x][y] += error

        if episode_count % 10000 == 0:
            print_states_num(states_value)
        episode_count += 1

def print_states_num(states_value):
    print("State values:")
    for line in states_value:
        for x in line:
            print("%.2f"% x, end="  ")
        print('\n')


#MONTECARLO(first_visit=False)
TD()