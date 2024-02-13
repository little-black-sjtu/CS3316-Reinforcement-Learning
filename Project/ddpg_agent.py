from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
from tqdm import tqdm
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_size, action_size, 
    hidden_layer=300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, action_size)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))

        return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size, 
    hidden_layer1=300, hidden_layer2=400):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size+action_size, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc3 = nn.Linear(hidden_layer2, 1)


    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.1, sigma=.1, sigma_min = 0.05, sigma_decay=.99):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        """Resduce  sigma from initial value to min"""
        self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class DDPG_Agent:
    def __init__(self, env, args):
        self.args = args

        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        
        self.Actor = Actor(self.state_size, self.action_size).to(self.args.device)
        self.Actor_target = Actor(self.state_size, self.action_size).to(self.args.device)
        if self.args.test_ddpg:
            if self.args.model != None:
                pth_actor = f"{self.args.model}_actor.pth"
            else:
                pths = os.listdir(self.args.save_dir)
                pth_actor = pths[-2]
                
        else:
            self.Actor.load_state_dict(torch.load(os.path.join(self.args.save_dir, pth_actor)))
            
        self.Actor_target.load_state_dict(self.Actor.state_dict())
        self.optimizer_actor = torch.optim.Adam(self.Actor.parameters(), lr=self.args.lr)

        self.Critic = Critic(self.state_size, self.action_size).to(self.args.device)
        self.Critic_target = Critic(self.state_size, self.action_size).to(self.args.device)
        if self.args.test_ddpg:
            if self.args.model != None:
                pth_critic = f"{self.args.model}_critic.pth"
            else:
                pths = os.listdir(self.args.save_dir)
                pth_critic = pths[-1]
        else:
            self.Critic.load_state_dict(torch.load(os.path.join(self.args.save_dir, pth_critic)))
            
        self.Critic_target.load_state_dict(self.Critic.state_dict())
        self.optimizer_critic = torch.optim.Adam(self.Critic.parameters(), lr=self.args.lr / 10)

        self.batch_size = self.args.batch_size

        self.memory = ReplayMemory(self.batch_size, self.args.device)
        self.tau = self.args.tau
        self.gamma = self.args.gamma
        
        self.noise = OUNoise(self.action_size, 0)
        
        self.current_update_n = 0
    
    def save_transition(self, state, action, done, reward, next_state):
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        done = torch.tensor(done)
        
        self.memory.push(state, next_state, action, reward, done)

    def make_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.args.device)
        action = self.Actor(state)
        action = action.cpu().detach().numpy().flatten()
        
        action += self.noise.sample()
        return action 

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0

        STATE, ACTION, REWARD, NEXT_STATE, DONE = self.memory.sample(self.batch_size)
        # transition = state, action, reward, next_state, done
    
        y = REWARD + (1-DONE) * self.gamma * self.Critic_target(NEXT_STATE, self.Actor_target(NEXT_STATE)).detach()
        Q = self.Critic(STATE, ACTION)

        critic_loss = F.mse_loss(Q, y)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        actor_loss = torch.mean(- self.Critic(STATE, self.Actor(STATE)))

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        for param, target_param in zip(self.Critic.parameters(), self.Critic_target.parameters()):
            target_param.data = self.tau * param.data + (1 - self.tau) * target_param.data
        for param, target_param in zip(self.Actor.parameters(), self.Actor_target.parameters()):
            target_param.data = self.tau * param.data + (1 - self.tau) * target_param.data
            
        self.current_update_n += 1
        
        return critic_loss.item() + actor_loss.item()
    
    def test(self):
        pbar = tqdm(range(self.args.num_episodes))

        total_rewards = 0.0
        total_len = 0

        for episode in pbar:
            obs, _ = self.env.reset()

            ep_reward = 0.0

            for s in range(self.args.max_num_steps):
                action = self.make_action(obs)
                next_obs, reward, done, _, _ = self.env.step(action)

                ep_reward += reward
                obs = next_obs

                if done:
                    break
            
            total_rewards += ep_reward
            total_len += s
            
        print("avg_rewards: %.4f, avg_length: %.4f"%(total_rewards / self.args.num_episodes, total_len / self.args.num_episodes))
        
    def train(self):
        pbar = tqdm(range(self.args.num_episodes))
        
        log_file = open(self.args.log_file, "a")
        log_file.write("Episode, Step, Reward, Loss, Length\n")
        
        total_loss = 0.0
        total_rewards = 0.0
        total_len = 0
        current_loss = 0.0

        for episode in pbar:
            obs, _ = self.env.reset()

            ep_loss = 0.0
            ep_reward = 0.0

            for s in range(self.args.max_num_steps):
                action = self.make_action(obs)
                next_obs, reward, done, _, _ = self.env.step(action)
                self.save_transition(obs, action, done, reward, next_obs)

                ep_loss += self.learn()

                ep_reward += reward

                if (self.current_update_n+1) % self.args.saver_steps == 0:
                    path = f"{self.args.save_dir}/{self.current_update_n+1}_actor.pth"
                    torch.save(self.Actor.state_dict(), path)
                    path = f"{self.args.save_dir}/{self.current_update_n+1}_critic.pth"
                    torch.save(self.Critic.state_dict(), path)
                
                obs = next_obs

                if done:
                    break
            
            total_loss += ep_loss
            total_rewards += ep_reward
            total_len += s
            current_loss = ep_loss

            if (episode+1) % self.args.num_eval == 0:
                avg_loss = total_loss / float(self.args.num_eval)
                avg_rewards = total_rewards / float(self.args.num_eval)
                avg_len = total_len / float(self.args.num_eval)

                total_loss = 0
                total_rewards = 0
                total_len = 0

                log_file.write("%d, %d, %.2f, %.4f, %.2f\n"%(episode, self.current_update_n, avg_rewards, avg_loss, avg_len))
                log_file.flush()
            
            pbar.set_description("sigma: %.2f, loss: %.4f"%(self.noise.sigma, current_loss))

class ReplayMemory:

    def __init__(
            self,
            capacity: int,
            device: str,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 0

        sink = lambda x: x.to(device)
        self.__m_states = sink(torch.zeros((capacity, 11), dtype=torch.float32))
        self.__m_next_states = sink(torch.zeros((capacity, 11), dtype=torch.float32))
        self.__m_actions = sink(torch.zeros((capacity, 3), dtype=torch.float32))
        self.__m_rewards = sink(torch.zeros((capacity, 1), dtype=torch.float32))
        self.__m_dones = sink(torch.zeros((capacity, 1), dtype=torch.bool))

    def push( self, state, next_state, action, reward, done):
        self.__m_states[self.__pos] = state
        self.__m_next_states[self.__pos] = next_state
        self.__m_actions[self.__pos] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done

        self.__pos += 1
        self.__size = max(self.__size, self.__pos)
        self.__pos %= self.__capacity

    def sample(self, batch_size: int):

        indices = torch.randint(0, high=self.__size, size=(batch_size,))
        b_state = self.__m_states[indices].to(self.__device)
        b_next = self.__m_next_states[indices].to(self.__device)
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device)
        b_done = self.__m_dones[indices].to(self.__device).float()
        return b_state, b_action, b_reward, b_next, b_done

    def __len__(self) -> int:
        return self.__size