import gymnasium as gym
import numpy as np
from MountaincarAgent import MountaincarAgent
from tqdm import tqdm
import pickle

def train():
    env = gym.make("MountainCar-v0")
    n_episodes = 500
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

    agent = MountaincarAgent(env.observation_space.shape[0], 3, use_double_dqn=False)

    with tqdm(range(0, n_episodes)) as pbar:
        best_reward = -200
        for i in pbar:
            obs, info = env.reset()
            done = False

            total_reward = 0.0
            losslist = []
            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.store_transition(obs, action, reward, next_obs, done)
                
                if agent.ready_for_train():
                    loss = agent.update_model()
                    losslist.append(loss)

                obs = next_obs
                total_reward += reward
            
            avg_loss = np.mean(np.array(losslist))
            agent.losslist.append(avg_loss)
            if total_reward > best_reward:
                best_reward = total_reward
                with open(f"{str(i+1).zfill(5)}with{best_reward}.pth", 'wb') as f:
                    pickle.dump(agent, f)
            
            pbar.set_postfix(
                {
                    "reward": total_reward,
                    "best_reward": best_reward,
                    "epsilon": agent.epsilon,
                    "avg_loss": avg_loss
                }
            )

            if i == n_episodes - 1:
                with open(f"{str(i+1).zfill(5)}end.pth", 'wb') as f:
                    pickle.dump(agent, f)

def test_visual():
    env = gym.make("MountainCar-v0", render_mode="human")
    with open("230ep_ddqn.pth", 'rb') as f:
        agent = pickle.load(f)
    
    obs, info = env.reset()
    done = False

    total_time_step = 0
    total_score = 0
    while not done:
        env.render()
        action = agent.get_action(obs, train=False)

        print('<--' if action == 0 else ('-->' if action == 2 else 'stop'))

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs = next_obs

        total_time_step += 1
        total_score += reward

    print(f"Total time: {total_time_step} \n Total score: {total_score}")

if __name__ == "__main__":
    test_visual()