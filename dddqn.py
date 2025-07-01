import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import random



class ReplayMemory():

    def __init__(self, memory_capacity=10000):
        self.buffer = deque(memory_capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.stack(states), actions, rewards, np.stack(next_states), dones

    def __len__(self):
        return len(self.buffer)
    

# low key diners drive ins and dives ah q-network

class DDDQN(nn.Module):

    # TODO: make action dim (5, 5)
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.layer1 = torch.nn.Linear(state_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.advantage = torch.nn.Linear(hidden_dim, action_dim)
        self.value = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Forward pass. Two linear layers, following by the dueling split layer
        (into advantage and value). Returns our Q value update.
        """
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        advantage = self.advantage(x)
        value = self.value(x)

        return value + (advantage - advantage.mean(dim=1, keepdim=True)[0])
    
    def advantage(self, state):
        """
        Forward pass. Only returns the advantage (use for action selection).
        """
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        advantage = self.advantage(x)

        return advantage
    


# TODO: ensure this interfaces well with OpenAI gym. I think it should.
class DDDQN_Agent():

    def __init__(self, state_size, action_size, hidden_size=64, gamma=0.99, replace=100, lr=0.001, memory_capacity=10000, batch_size=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size

        self.main_network = DDDQN(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=self.gamma) # only for main network!

        self.target_network = DDDQN(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()

        self.memory = ReplayMemory(memory_capacity)


    def learn(self, state, action, reward, next_state, done):
        """ Store tuple in memory """
        self.memory.store(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            self.update_model()


    def select_action(self, state, eps=0):
        """ Epsilon-greedy action selection """
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.main_network.eval()
            with torch.no_grad():
                # Q-values corresponding to each action
                action_values = self.main_network(state)

            self.main_network.train()
            return np.argmax(action_values.cpu().data.numpy())
        
        else:
            return random.choice(np.arange(self.action_size))
        

    def update_model(self):
        """ Update model based on loss between expected Q values (from DDDQN) and actual Q values """
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size) # stochastic

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().to(self.device)

        q_values = self.main_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        next_action_values = self.main_network(next_states).max(1)[1].unsqueeze(-1)
        next_q_values = self.target_network(next_states).gather(1, next_action_values).detach().squeeze(-1)

        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = torch.nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def hard_update(self):
        """ Periodically update target network parameters to match main network """
        self.target_network.load_state_dict(self.main_network.state_dict())
