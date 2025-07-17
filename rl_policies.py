### RL algorithms to learn optimal policies using some flavor of Deep Q-Networks ###

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
from tqdm import tqdm



def train_network(self, env, n_episodes=50_000, target_network=False, update_freq=1000):
    """
    Training loop to learn a Q-function.
    """
    rewards, lengths = [], []

    for i in tqdm(range(n_episodes)):
        state, _ = env.reset()
        done = False
        total_reward, episode_length = 0, 0

        while not done:
            action = self.select_action(torch.from_numpy(state))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # store tuple in memory and do one training step
            self.learn(state, action, reward, next_state, done)
            # update for next timestep
            state = next_state
            total_reward += reward
            episode_length += 1

        rewards.append(total_reward)
        lengths.append(episode_length)

        # update the target network weights to match main network
        if target_network and i % update_freq == 0:
            self.hard_update()

    return rewards, lengths



class PPO():
    """
    Proximal Policy Optimization. A policy gradient method.
    """
    pass



class ReplayMemory():

    def __init__(self, memory_capacity=100_000):
        self.buffer = deque(maxlen=memory_capacity)

    def store(self, state, action, reward, next_state, done):
        """
        Stores a (s_t, a, r, s_t+1, d) tuple in memory.
        """
        state = torch.from_numpy(state) # np array --> tensor
        next_state = torch.from_numpy(next_state) # np array --> tensor
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        """
        Samples a batch of (s_t, a, r, s_t+1, d) tuples from memory.
        Returns values as tensors.
        """
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))

        states = torch.stack(list(states), dim=0).float().to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.stack(list(next_states), dim=0).float().to(device)
        dones = torch.tensor(dones, dtype=torch.uint8).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    

class DDQN(nn.Module):
    """ Dueling Deep Q-Network! """

    # TODO: add batch normalization, leaky relu (from raghu, et al.)
    def __init__(self, state_dim, action_dim, hidden_dim):
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

        return value + (advantage - advantage.mean()) # (25,)




class DDQNAgent():
    """
    Dueling Deep Q-Network Agent

    This is a functional agent by itself, but it's also a base class from which we can
    implement D3QN and CQL (with just a few modifications to the Q-function update).
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.99, lr=0.001, 
                 memory_capacity=100_000, batch_size=128, eps_start=0.1, eps_decay=0.9999, eps_final=0.05):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim = state_dim # 20, i.e., (20,) latent vector
        self.action_dim = action_dim # 25, i.e., 1 q-value per discrete action tuple
        self.gamma = gamma
        self.batch_size = batch_size

        self.eps = eps_start
        self.eps_decay = eps_decay
        self.eps_final = eps_final

        self.main_network = DDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=lr)

        self.memory = ReplayMemory(memory_capacity)


    def learn(self, state, action, reward, next_state, done):
        """ Store tuple in memory and do one round of model updates. """
        self.memory.store(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            self.main_network.train()
            self.update_model()


    def get_action_values(self, state):
        """ Returns a Tensor, with the learned Q-value associated with each action. """
        self.main_network.eval()
        with torch.no_grad():
            return self.main_network(state)
            

    def get_action_probs(self, state):
        """ Returns a probability distribution over the action space. """
        action_values = self.get_action_values(state)
        distribution = F.softmax(action_values, dim=-1)
        return distribution.detach().numpy()


    def select_action(self, state, eps=None):
        """ 
        With probability 1-eps, selects action with the highest corr. Q value from the network. 
        With probability eps, selects an action at random.
        """
        # choose a random action, with probability eps
        if eps is None:
            eps = self.anneal_epsilon()
        if np.random.rand() < eps:
            return np.random.randint(self.action_dim)
        
        # choose the action that maximizes our q-function, with probability (1-eps)
        action_values = self.get_action_values(state)
        return torch.argmax(action_values).item() # in [0, 24]
    

    def anneal_epsilon(self):
        """
        Exponentially decays epsilon until it reaches a min. value.
        """
        self.eps = max(self.eps * self.eps_decay, self.eps_final)
        return self.eps
        

    def update_model(self):
        """
        Update model based on loss between expected Q values (from DDDQN) and actual Q values.
        """
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device) # stochastic

        q_values = self.main_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_action_values = self.main_network(next_states).max(1)[1].unsqueeze(-1)
        next_q_values = self.main_network(next_states).gather(1, next_action_values).detach().squeeze(-1)

        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = torch.nn.MSELoss()(q_values, expected_q_values) # TODO: try Huber loss?
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



# low key diners drive ins and dives ah q-network

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
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device) # stochastic

        self.main_network.train()
        q_values = self.main_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_action_values = self.main_network(next_states).max(1)[1].unsqueeze(-1)
        # NOTE: target network!!
        next_q_values = self.target_network(next_states).gather(1, next_action_values).detach().squeeze(-1)

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
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device) # stochastic

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
