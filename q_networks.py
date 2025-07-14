### RL algorithms to learn optimal policies using some flavor of Deep Q-Networks ###

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
    """ Dueling Deep Q-Network! """

    # TODO: add batch normalization, leaky relu (from raghu, et al.)
    def __init__(self, state_dim, action_dim, hidden_dim, dueling=True):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.advantage = torch.nn.Linear(hidden_dim, action_dim)
        self.value = torch.nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """
        Forward pass. Two linear layers, following by the dueling split layer
        (into advantage and value). Returns our Q value update.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        advantage = self.advantage(x)
        value = self.value(x)

        return value + (advantage - advantage.mean(dim=1, keepdim=True)) # (25,)




class DDQNAgent():
    """
    Dueling Deep Q-Network Agent

    This is a functional agent by itself, but it's also a base class from which we can
    implement D3QN and CQL (with just a few modifications to the Q-function update).
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64, gamma=0.99, lr=0.001, 
                 memory_capacity=10000, batch_size=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim = state_dim # 20, i.e., (20,) latent vector
        self.action_dim = action_dim # 25, i.e., 1 q-value per discrete action tuple
        self.gamma = gamma
        self.batch_size = batch_size

        self.main_network = DDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=lr)

        self.memory = ReplayMemory(memory_capacity)


    def learn(self, state, action, reward, next_state, done):
        """ Store tuple in memory. """
        self.memory.store(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            self.main_network.train()
            self.update_model() # this is the main training loop


    def get_action_values(self, state):
        """ Returns the learned Q-value associated with each action. """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.main_network.eval()
        with torch.no_grad():
            # Q-values corresponding to each action
            action_values = self.main_network(state)

        return action_values.cpu().data.numpy() # np array (25,)
    

    def get_action_probs(self, state):
        """ Returns a probability distribution over the action space. """
        action_values = self.get_action_values(state)
        return F.softmax(action_values)


    def select_action(self, state):
        """ Greedily selects action with the highest corr. Q value from the network. """
        action_values = self.get_action_values(state)
        return np.argmax(action_values) # in [0, 24]
        

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
        next_q_values = self.main_network(next_states).gather(1, next_action_values).detach().squeeze(-1)

        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = torch.nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



class D3QNAgent(DDQNAgent):
    """ Dueling Double Deep Q-Network Agent. adds in target network """

    def __init__(self, state_dim, action_dim, hidden_dim=64, gamma=0.99, lr=0.001, 
                 memory_capacity=10000, batch_size=64):
        super().__init__(state_dim, action_dim, hidden_dim, gamma, lr, memory_capacity, batch_size)

        self.target_network = DDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict()) # same as main_network @init
        self.target_network.eval()
        

    def update_model(self):
        """ Update model based on loss between expected Q values (from DDDQN) and actual Q values """
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size) # stochastic

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(np.array(dones)).float().to(self.device)

        q_values = self.main_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        next_action_values = self.main_network(next_states).max(1)[1].unsqueeze(-1)
        # using bellman backups
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
    """
    Essentially DDQN, but performs a Q value update for which the expectation is a lower bound of the actual Q value!
    Should be careful that I don't underestimate too far. Using dueling, but not double, for this reason.
    """

    def cql_loss(self, q_values, current_action):
        """ Computes the CQL loss for a batch of Q-values. Penalizes action values not observed in data. """
        logsumexp = torch.logsumexp(q_values, dim=-1)
        q_a = q_values.gather(-1, current_action)
        return (logsumexp - q_a).mean()
    

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
        # using bellman backups
        next_q_values = self.main_network(next_states).gather(1, next_action_values).detach().squeeze(-1)

        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        cql1_loss = self.cql_loss(q_values, actions) # NOTE: extra loss term
        td_loss = torch.nn.MSELoss()(q_values, expected_q_values)
        q1_loss = cql1_loss + 0.5 * td_loss # TODO: can tune weighting

        self.optimizer.zero_grad()
        q1_loss.backward()
        self.optimizer.step()



class FQIAgent(D3QNAgent):
    """
    Uses Fitted-Q Evaluation (FQE) to approximate the clinician Q-function using a neural net.
     
    TODO: what are we actually doing here? Is this imitation learning / behavior cloning?
    """

    def update_model(self):
        """ Supervised Q-learning: regress Q(s, a) to observed reward (clinician action). """
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)  # ignore next_state, done

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(np.array(dones)).float().to(self.device)

        q_values = self.main_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + self.gamma * (1 - dones) * max_next_q

        loss = torch.nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()