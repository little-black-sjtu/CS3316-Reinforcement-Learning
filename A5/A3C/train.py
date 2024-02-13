import gymnasium as gym
import numpy as np
from PendulumAgent import A3C_PendulumAgent
from tqdm import tqdm
import torch.multiprocessing as mp
import pickle

def multiprocess_train():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    n_episodes = 2000

    agent = A3C_PendulumAgent(state_dim, action_dim, device='cuda')

    with tqdm(range(0, n_episodes)) as pbar:        
        best_reward = -3200
        for i in pbar:
            worker_reward = 0
            for worker in range(agent.num_workers):
                state, _= env.reset()
                done = False

                while not done:
                    action = agent.choose_action(state)
                    action_scaled = action * max_action

                    next_state, reward, terminated, truncated, _ = env.step(action_scaled)
                    done = terminated or truncated

                    agent.store_transition(state, action, reward, next_state, done)

                    state = next_state
                    worker_reward += reward
            agent.update()

            ep_reward = worker_reward / agent.num_workers
            agent.reward_list.append(ep_reward)
            if ep_reward > best_reward:
                best_reward = ep_reward

            pbar.set_postfix(
                {
                    'ep_reward' : ep_reward,
                    'best_reward' : best_reward
                }
            )

            if (i+1)%50 == 0:
                with open(f"{str(i+1).zfill(5)}epoch.pth", 'wb') as f:
                    pickle.dump(agent, f)



if __name__ == "__main__":
    multiprocess_train()