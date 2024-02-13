from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
from collections import deque

class Q_net(nn.Module):
    def __init__(self, state_size, action_size, units=64):
        super(Q_net, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, action_size)
        )
        self.lossfunction = nn.SmoothL1Loss()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    
    def forward(self, x): 
        return self.network(x)

    def predict(self, x):
        pred = torch.softmax(self.forward(x), dim=1)
        return torch.argmax(pred, dim=1)

    def getloss(self, y_pre, y_target):
        return self.lossfunction(y_pre, y_target)

class MountaincarAgent:

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate: float = 0.0005,
        total_length_memory: int = 2000,
        train_start_length_memory: int = 100,
        updating_batch_size: int = 64,
        discount_factor: float = 0.98,
        target_update_interval: int = 10,
        epsilon_start = 1.0,
        epsilon_decay = 20000,
        epsilon_end = 0.05,
        use_double_dqn = True,
    ):
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.state_size = state_size
        self.action_size = action_size
        
        self.action_q_model = Q_net(self.state_size, self.action_size)
        self.target_q_model = Q_net(self.state_size, self.action_size)
        self.target_q_model.load_state_dict(self.action_q_model.state_dict())

        self.memory = deque(maxlen=total_length_memory)
        self.train_start_length_memory = train_start_length_memory

        self.updating_batch_size = updating_batch_size
        self.optimizer = optim.Adam(self.action_q_model.parameters(), lr=self.lr)

        self.target_update_interval = target_update_interval
        self.current_update_n = 0

        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
    
        self.use_double_dqn = use_double_dqn
        self.losslist = []

    def ready_for_train(self):
        return len(self.memory) >= self.train_start_length_memory
    
    def get_action(self, obs, train=True):
        if np.random.random() < self.epsilon and train:
            return random.randrange(self.action_size)

        else:
            return self.action_q_model.predict(torch.tensor(np.reshape(obs, (1,2)))).item()
    
    def store_transition(self, obs, action, reward, next_obs, done):
        transition = (obs, action, reward, next_obs, done)
        self.memory.append(transition)

        self.epsilon = (self.epsilon - self.epsilon_end) * np.exp(- 1 / self.epsilon_decay) + self.epsilon_end

    def traditional_predict(self, update_input, update_target, action, done, reward):
        action = torch.tensor(np.reshape(np.array(action, dtype=np.int64), (-1,1)))

        predict_val = self.action_q_model(update_input).gather(1, action)
        target_val = self.target_q_model(update_target).max(1, keepdim=True)[0].detach()

        reward = np.array(reward)
        reward = torch.unsqueeze(torch.tensor(reward), 1)
        done = (np.array(done)==False)
        done = torch.unsqueeze(torch.tensor(done, dtype=torch.int32), 1)
        discount_factor = torch.tensor(self.discount_factor)

        target_val = reward + done * discount_factor * target_val
        
        return predict_val, target_val
    
    def double_predict(self, update_input, update_target, action, done, reward):
        action = torch.tensor(np.reshape(np.array(action, dtype=np.int64), (-1,1)))

        predict_val = self.action_q_model(update_input).gather(1, action)
        target_val_query = self.action_q_model(update_target).max(1, keepdim=True)[1]
        target_val = self.target_q_model(update_target).gather(1, target_val_query)

        reward = np.array(reward)
        reward = torch.unsqueeze(torch.tensor(reward), 1)
        done = (np.array(done)==False)
        done = torch.unsqueeze(torch.tensor(done, dtype=torch.int32), 1)
        discount_factor = torch.tensor(self.discount_factor)

        target_val = reward + done * discount_factor * target_val
        return predict_val, target_val

    def update_model(self):
        batch_size = self.updating_batch_size
        transitions = random.sample(self.memory, batch_size)
        # transition = [phi_j, a, r_j, phi_j+1, done]
        update_input = np.zeros((batch_size, self.state_size), dtype=np.float32)
        update_target = np.zeros((batch_size, self.state_size), dtype=np.float32)
        action, reward, done = [], [], []

        for i, transition in enumerate(transitions):
            update_input[i] = transition[0]
            action.append(transition[1])
            reward.append(transition[2])
            update_target[i] = transition[3]
            done.append(transition[4])

        update_input = torch.tensor(update_input)
        update_target = torch.tensor(update_target)

        update_input = update_input
        update_target = update_target

        if self.use_double_dqn:
            predict_val, target_val = self.double_predict(update_input, update_target, action, done, reward)
        else:
            predict_val, target_val = self.traditional_predict(update_input, update_target, action, done, reward)

        loss = self.action_q_model.getloss(predict_val, target_val)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.current_update_n += 1
        if self.current_update_n % self.target_update_interval == 0:
            self.target_q_model.load_state_dict(self.action_q_model.state_dict())
        
        return loss.item()