import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
from tqdm import tqdm
import time
import os

class DQN_Agent:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        self.batch_size = self.args.batch_size
        self.lr = self.args.lr
        self.gamma = self.args.gamma  # reward decay rate

        self.action_dim = env.action_space.n
        self.device = self.args.device

        self.epsilon = self.args.epsilon_start

        if self.args.test_dqn or self.args.keep_train:
            if self.args.model != None:
                pth = self.args.model + ".pth"
            else:
                pths = os.listdir(self.args.save_dir)
                pth = pths[-1]
            self.policy_net = DQN(action_dim=self.action_dim).to(self.device)
            self.target_net = DQN(action_dim=self.action_dim).to(self.device)

            self.policy_net.load_state_dict(torch.load(os.path.join(self.args.save_dir, pth)))
            self.policy_net.load_state_dict(torch.load(os.path.join(self.args.save_dir, pth)))
            

            self.epsilon = 0
            if self.args.keep_train:
                self.epsilon = 0.5
        else:
            self.policy_net = DQN(action_dim=self.action_dim).to(self.device)
            self.target_net = DQN(action_dim=self.action_dim).to(self.device)
            self.policy_net.apply(DQN.init_weights)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.lr,
        )
        
        self.memory = ReplayMemory(self.args.memory_size, self.device)

        self.current_update_n = 0

    def make_action(self, obs):
        obs = torch.FloatTensor(np.array(obs) / 255.0).unsqueeze(0).to(self.device)
        q_values = self.policy_net(obs)
        

        if random.random() <= self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            action = np.argmax(q_values.cpu().detach().numpy())

        if self.epsilon > self.args.epsilon_end :
            self.epsilon -= (self.args.epsilon_start - self.args.epsilon_end) / float(self.args.explore_steps)
        
        return action

    def save_transition(self, obs, action, done, reward, next_obs):
        # self.memory.push((np.array(obs), int(action), done, int(reward), np.array(next_obs)))
        obs = np.array(obs)
        next_obs = np.array(next_obs)
        state = np.concatenate((obs, next_obs[[3], ...]))
        state = torch.tensor(state, dtype=torch.uint8)

        # action = torch.Tensor(action, dtype=torch.long)
        # reward = torch.Tensor(reward, dtype=torch.int8)
        # done = torch.IntTensor(done, dtype=torch.bool)

        self.memory.push(state, action ,reward, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        obs, action, reward, next_obs, done = self.memory.sample(self.batch_size)
        
        predict = self.policy_net(obs).gather(1, action)
        target = reward + self.gamma * (1 - done) * self.target_net(next_obs).detach().max(1).values.detach().unsqueeze(1)

        loss = F.smooth_l1_loss(predict, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.current_update_n += 1
        if self.current_update_n % self.args.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

    def test(self):
        pbar = tqdm(range(self.args.num_episodes))

        total_rewards = 0.0
        total_len = 0

        for episode in pbar:
            obs, _ = self.env.reset()

            ep_reward = 0.0

            for s in range(self.args.max_num_steps):
                action = self.make_action(obs)
                if s == 0:
                    action = 1
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
        log_file.write("Episode, Step, Epsilon, Reward, Loss, Length\n")
        
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
                    path = f"{self.args.save_dir}/{self.current_update_n+1}.pth"
                    torch.save(self.policy_net.state_dict(), path)
                
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

                log_file.write("%d, %d, %.4f, %.2f, %.4f, %.2f\n"%(episode, self.current_update_n, self.epsilon, avg_rewards, avg_loss, avg_len))
                log_file.flush()
            
            pbar.set_description("epsilon: %.4f, loss: %.4f"%(self.epsilon, current_loss))

class DQN(nn.Module):
    '''All the image inputs are scaled to 84*84 grayscale, and 4 preprocessed images are stacked'''

    def __init__(self, action_dim):
        super(DQN, self).__init__()
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.__fc1 = nn.Linear(64*7*7, 512)
        self.__fc2 = nn.Linear(512, action_dim)
    
    def forward(self, x):
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        x = F.relu(self.__fc1(x.view(x.size(0), -1)))
        return self.__fc2(x)

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

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
        self.__m_states = sink(torch.zeros((capacity, 5, 84, 84), dtype=torch.uint8))
        self.__m_actions = sink(torch.zeros((capacity, 1), dtype=torch.long))
        self.__m_rewards = sink(torch.zeros((capacity, 1), dtype=torch.int8))
        self.__m_dones = sink(torch.zeros((capacity, 1), dtype=torch.bool))

    def push( self, folded_state, action, reward, done):
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done

        self.__pos += 1
        self.__size = max(self.__size, self.__pos)
        self.__pos %= self.__capacity

    def sample(self, batch_size: int):

        indices = torch.randint(0, high=self.__size, size=(batch_size,))
        b_state = self.__m_states[indices, :4].to(self.__device).float() / 255.
        b_next = self.__m_states[indices, 1:].to(self.__device).float()  / 255.
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        return b_state, b_action, b_reward, b_next, b_done

    def __len__(self) -> int:
        return self.__size