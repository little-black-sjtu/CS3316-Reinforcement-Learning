import copy

grid_size = [6, 6]

terminal = [[0, 1],[5, 5]]


actions = [[-1, 0], #north
           [0, +1], #east
           [+1, 0], #south
           [0, -1]] #west

direction_names = ['↑', '→', '↓', '←']

def get_next_state_value_for_action(states_value, state, action):
    x, y = state
    dx, dy = action
    new_x = x + dx
    new_y = y + dy
    x = x if new_x < 0 or new_x >= grid_size[0] else new_x
    y = y if new_y < 0 or new_y >= grid_size[1] else new_y
    return states_value[x][y]

def is_terminal(state, terminal):
    for t in terminal:
        if state[0] == t[0] and state[1] == t[1]:
            return True
    return False
    
def iterate_one_time(states_value, policy, penalty = -1, converge_num = 10e-5):
    origin_states_values = copy.deepcopy(states_value)
    delta = 0.0
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            state = [x, y]
            if is_terminal(state, terminal):
                continue
            
            new_state_value = 0.0
            state_policy = policy[x][y]
            for act in range(len(actions)):
                next_state_value = get_next_state_value_for_action(origin_states_values, state, actions[act])
                new_state_value += state_policy[act]*(penalty + next_state_value)
            
            delta = max(delta, abs(origin_states_values[x][y] - new_state_value)) 
            states_value[x][y] = new_state_value

    return delta < converge_num

def get_states_value_of_policy(states_value, policy, iter_times = 10000):

    iter_time = 0
    while iter_time < iter_times:
        if iterate_one_time(states_value, policy):
            break
        iter_time += 1
    
    return states_value

def get_new_policy_by_old_policy(states_value, policy, converge_num = 10e-3):
    old_policy = copy.deepcopy(policy)
    delta = 0
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            state = [x, y]
            state_value_four_directions = []

            for action in actions:
                state_value_four_directions.append(get_next_state_value_for_action(states_value, state, action))

            maximal_reward = max(state_value_four_directions)

            count = 0
            for i in range(len(state_value_four_directions)):
                if round(maximal_reward, 1) == round(state_value_four_directions[i], 1):
                    policy[x][y][i] = 1
                    count += 1
                else:
                    policy[x][y][i] = 0
            for i in range(len(state_value_four_directions)):
                if policy[x][y][i] == 1:
                    policy[x][y][i] /= count
                delta = max(delta, policy[x][y][i] - old_policy[x][y][i])
    print_policy_arrow(policy)
    return delta < converge_num

def print_states_num(states_value):
    print("State values:")
    for line in states_value:
        for x in line:
            print("%.2f"% x, end="  ")
        print('\n')

def print_policy_arrow(policy):
    print("Policy:")
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            if is_terminal([x, y], terminal):
                print("T   ", end="")
                continue
            for i in range(len(actions)):
                if policy[x][y][i] > 0:
                    print(f"{direction_names[i]}", end="")
                else:
                    print("", end="")
            print("   ", end="")
        print('\n')

def print_policy_num(policy):
    print("Policy:")
    for x in policy:
        for y in x:
            for i in y:
                print(i, end=" ")
            print("   ", end="")
        print('\n')

def policy_iteration(policy):
    states_value = [[0 for i in range(grid_size[1])] for i in range(grid_size[0])]
    while True:
        states_value = get_states_value_of_policy(states_value, policy)
        print_states_num(states_value)
        if get_new_policy_by_old_policy(states_value, policy):
            break

def value_iteration(policy):
    states_value = [[0 for i in range(grid_size[1])] for i in range(grid_size[0])]
    # the only difference is that the step for value_iteration is 1
    while True:
        states_value = get_states_value_of_policy(states_value, policy, iter_times=1) 
        print_states_num(states_value)
        if get_new_policy_by_old_policy(states_value, policy):
            break

policy = [[[0.25 for i in range(len(actions))]for i in range(grid_size[1])]for i in range(grid_size[0])] # initialize the policy with all 0.25
policy_iteration(policy)
