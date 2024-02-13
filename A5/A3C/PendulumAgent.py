from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_size, action_size, 
    hidden_layer1=400, hidden_layer2=300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_layer1)
        self.uniform_init(self.fc1, 1 / np.sqrt(self.fc1.weight.data.size()[0]))

        self.bn1 = nn.LayerNorm(hidden_layer1)

        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.uniform_init(self.fc2, 1 / np.sqrt(self.fc2.weight.data.size()[0]))

        self.bn2 = nn.LayerNorm(hidden_layer2)

        self.fc3 = nn.Linear(hidden_layer2, action_size)
        self.uniform_init(self.fc3, 0.003)

        

    def uniform_init(self, layer, thelshold):
        nn.init.uniform_(layer.weight.data, -thelshold, thelshold)
        nn.init.uniform_(layer.bias.data, -thelshold, thelshold)
    
    def forward(self, x): 
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size, 
    hidden_layer1=400, hidden_layer2=300):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_layer1)
        self.uniform_init(self.fc1, 1 / np.sqrt(self.fc1.weight.data.size()[0]))

        self.bn1 = nn.LayerNorm(hidden_layer1)

        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.uniform_init(self.fc2, 1 / np.sqrt(self.fc2.weight.data.size()[0]))

        self.fc_action = nn.Linear(action_size, hidden_layer2)

        self.bn2 = nn.LayerNorm(hidden_layer2)

        self.fc3 = nn.Linear(hidden_layer2, 1)
        self.uniform_init(self.fc3, 0.003)

    def uniform_init(self, layer, thelshold):
        nn.init.uniform_(layer.weight.data, -thelshold, thelshold)
        nn.init.uniform_(layer.bias.data, -thelshold, thelshold)

    def forward(self, s, a):
        s = self.fc1(s)
        s = self.bn1(s)
        s = F.relu(s)
        s = self.fc2(s)
        s = self.bn2(s)
        a = self.fc_action(a)

        x = F.relu(torch.add(s, a))
        x = self.fc3(x)
        return x

class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu, sigma=0.2, theta=0.15):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev)  + self.sigma * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = np.zeros_like(self.mu)

class A3C_PendulumAgent:
    # DDPG
    def __init__(
        self,
        state_size,
        action_size,
        lr_actor: float = 0.0001,
        lr_critic: float = 0.001,
        gamma: float = 0.9,
        device: str = 'cuda',
        train: bool = True,
        num_workers: int = 5,
    ):
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.state_size = state_size
        self.action_size = action_size
        
        self.Global_Actor = Actor(state_size, action_size).to(device)
        self.optimizer_actor = torch.optim.Adam(self.Global_Actor.parameters(), lr=lr_actor)

        self.Global_Critic = Critic(state_size, action_size).to(device)
        self.optimizer_critic = torch.optim.Adam(self.Global_Critic.parameters(), lr=lr_critic)

        self.noise = OrnsteinUhlenbeckActionNoise(np.zeros(action_size))

        self.gamma = gamma

        self.num_workers = num_workers
        self.memory = []

        self.device = device
        self.train = train
        
        self.reward_list = []
    
    def store_transition(self, state, action, reward, next_state, done):
        transition = [state, action, reward, next_state, done]
        self.memory.append(transition)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.Global_Actor(state)
        if self.train:
            action = action + torch.FloatTensor(self.noise()).to(self.device)

        action = action.cpu().detach().numpy().flatten()
        return action 

    def update(self):
        # transition = state, action, reward, next_state, done
        STATE, ACTION, REWARD, NEXT_STATE, DONE = zip(*self.memory)

        STATE = torch.FloatTensor(STATE).to(self.device)
        ACTION = torch.FloatTensor(ACTION).to(self.device)
        NEXT_STATE = torch.FloatTensor(NEXT_STATE).to(self.device)

        REWARD = torch.FloatTensor(REWARD).unsqueeze(1).to(self.device)
        DONE = torch.Tensor(DONE).unsqueeze(1).to(self.device)

        
        y = REWARD + (1-DONE) * self.gamma * self.Global_Critic(NEXT_STATE, self.Global_Actor(NEXT_STATE)).detach()
        Q = self.Global_Critic(STATE, ACTION)

        critic_loss = F.mse_loss(Q, y)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        actor_loss = torch.mean(- self.Global_Critic(STATE, self.Global_Actor(STATE)))

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.memory = []
        
