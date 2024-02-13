import gymnasium as gym
import numpy as np
from PendulumAgent import A3C_PendulumAgent
import pickle
import matplotlib.pyplot as plt

def load_agent(name):
    with open(name, 'rb') as f:
        agent = pickle.load(f)

    return agent

def test_visual(agent, render_mode="human"):
    env = gym.make("Pendulum-v1", render_mode=render_mode)
    max_action = env.action_space.high[0]
    agent.train = False
    state, _ = env.reset()
    done = False

    total_score = 0
    while not done:
        if render_mode=='human':
            env.render()
        action = agent.choose_action(state)

        action_scaled = action * max_action

        next_state, reward, terminated, truncated, _ = env.step(action_scaled)
        done = terminated or truncated
        state = next_state

        total_score += reward

    print(f"Total reward: {total_score}")
    return total_score

def plot_rewardlist(agent):
    plt.plot(agent.reward_list)
    plt.show()

def cal_mean_score(agent, eval_time = 50):
    reward_list = []
    for i in range(eval_time):
        reward_list.append(test_visual(agent, render_mode='default'))
    print('Avg reward: ', np.mean(reward_list))

if __name__ == "__main__":
    name = "02000epoch.pth"
    agent = load_agent(name)
    # test_visual(agent)
    cal_mean_score(agent)
    plot_rewardlist(agent)
    