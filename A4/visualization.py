import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gymnasium as gym

def plt_action_heatmap(file_name):
    with open(file_name, 'rb') as f:
        agent = pickle.load(f)

    length = 30
    position = np.linspace(-1.2, 0.6, length)
    velocity = np.linspace(-0.07, 0.07, length)

    outputs = []
    for y in range(length):
        output = []
        for x in range(length):
            output.append(agent.get_action(np.array([position[x], velocity[y]], dtype=np.float32)))
        outputs.append(output)

    position = np.round(position, 2)
    velocity = np.round(velocity, 2)

    df = pd.DataFrame(outputs, index = velocity, columns = position)
    plt.figure(figsize=(8,8))
    sns.heatmap(df, cmap=sns.diverging_palette(10, 220, n=3), xticklabels=4, yticklabels=4, cbar=False, annot=True)

    plt.xlabel('Position of the Car')
    plt.ylabel('Velocity of the Car')
    plt.show()

def calculate_state_values(file_name):
    with open(file_name, 'rb') as f:
        agent = pickle.load(f)

    length = 30
    position = np.linspace(-1.2, 0.6, length)
    velocity = np.linspace(-0.07, 0.07, length)

    outputs = []
    for y in range(length):
        output = []
        for x in range(length):
            output.append(np.round(agent.action_q_model(torch.tensor([position[x], velocity[y]], dtype=torch.float32).view(1, 2)).max().item(), 0))
        outputs.append(output)

    position = np.round(position, 2)
    velocity = np.round(velocity, 2)

    df = pd.DataFrame(outputs, index = velocity, columns = position)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, xticklabels=4, yticklabels=4, annot=True)

    plt.xlabel('Position of the Car')
    plt.ylabel('Velocity of the Car')
    plt.show()


def plt_loss(file_name):
    with open(file_name, 'rb') as f:
        agent = pickle.load(f)
    plt.plot(agent.losslist)
    plt.show()

def calculate_avg_score(file_name, n=50):
    env = gym.make("MountainCar-v0")
    with open(file_name, 'rb') as f:
        agent = pickle.load(f)
    
    scores = []
    for i in range(n):
        obs, info = env.reset()
        done = False

        total_score = 0
        while not done:
            action = agent.get_action(obs, train=False)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs

            total_score += reward
        scores.append(total_score)
    print(f'avg_score: {np.mean(np.array(scores))}')

if __name__ == '__main__':
    file_name = '230ep_ddqn.pth'
    # file_name = '460ep_dqn.pth'
    # calculate_avg_score(file_name)
    calculate_state_values(file_name)