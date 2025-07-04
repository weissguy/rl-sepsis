import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import random



class ReplayMemory():

    def __init__(self, memory_capacity=10000):
        self.buffer = deque(maxlen=memory_capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.stack(states), actions, rewards, np.stack(next_states), dones

    def __len__(self):
        return len(self.buffer)
    

# low key diners drive ins and dives ah q-network

class DDQN(nn.Module):

    # TODO: add batch normalization, leaky relu, 'projection onto the action space' (from raghu, et al.)
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

        return value + (advantage - advantage.mean(dim=1, keepdim=True)) # (25,)
    
    


class DDQNAgent():

    def __init__(self, state_dim, action_dim, hidden_dim=64, gamma=0.99, lr=0.001, 
                 memory_capacity=10000, batch_size=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim = state_dim # 20, i.e., (20,) latent vector
        self.action_dim = action_dim # 25, i.e., 1 q-value per discrete action tuple
        self.gamma = gamma
        self.batch_size = batch_size

        self.main_network = DDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=lr) # only for main network!

        self.target_network = DDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict()) # same as main_network @init
        self.target_network.eval()

        self.memory = ReplayMemory(memory_capacity)


    def learn(self, state, action, reward, next_state, done):
        """ Store tuple in memory. """
        self.memory.store(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            self.update_model() # this is the main training loop


    def select_action(self, state):
        """ Greedily selects action with the highest corr. Q value from the network. """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.main_network.eval()
        with torch.no_grad():
            # Q-values corresponding to each action
            action_values = self.main_network(state)

        self.main_network.train()
        return np.argmax(action_values.cpu().data.numpy()) # in [0, 24]
        

    def update_model(self):
        """ Update model based on loss between expected Q values (from DDDQN) and actual Q values """
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size) # stochastic

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device) # TODO: why not list --> tensor?
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().to(self.device)

        q_values = self.main_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        next_action_values = self.main_network(next_states).max(1)[1].unsqueeze(-1)
        next_q_values = self.target_network(next_states).gather(1, next_action_values).detach().squeeze(-1) # NOTE: target network!!

        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = torch.nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def hard_update(self):
        """ Periodically update target network parameters to match main network """
        self.target_network.load_state_dict(self.main_network.state_dict())



class CQLAgent(DDQNAgent):
    """ Essentially DQN, but performs a Q value update for which the expectation is a lower bound of the actual Q value! """

    def cql_loss(self, q_values, current_action):
        """ Computes the CQL loss for a batch of Q-values. """
        logsumexp = torch.logsumexp(q_values, dim=-1)
        q_a = q_values.gather(-1, current_action)
        return (logsumexp - q_a).mean()
    

    def update_model(self):
        """ Update model based on loss between expected Q values (from DDDQN) and actual Q values """
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size) # stochastic

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device) # TODO: why not list --> tensor?
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().to(self.device)

        q_values = self.main_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        next_action_values = self.main_network(next_states).max(1)[1].unsqueeze(-1)
        next_q_values = self.target_network(next_states).gather(1, next_action_values).detach().squeeze(-1) # NOTE: target network!!

        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        cql1_loss = self.cql_loss(q_values, actions)
        td_loss = torch.nn.MSELoss()(q_values, expected_q_values)
        q1_loss = cql1_loss + 0.5 * td_loss

        self.optimizer.zero_grad()
        q1_loss.backward()
        self.optimizer.step()