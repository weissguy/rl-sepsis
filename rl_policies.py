### RL algorithms to learn optimal policies using some flavor of Deep Q-Networks ###

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
from collections import deque
import numpy as np
import random

from networks import DDQN, Actor, Critic, Value



class ReplayMemory():

    def __init__(self, memory_capacity=100_000):
        self.buffer = deque(maxlen=memory_capacity)

    def store(self, state, action, reward, next_state, done):
        """
        Stores a (s_t, a, r, s_t+1, d) tuple in memory.
        """
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
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)




class D3QNAgent():
    """
    Diners, Drive-ins, and Dives Q-Learning

    oh wait mb. Dueling Double Deep Q-Learning
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.99, lr=1e-4,
                 memory_capacity=100_000, batch_size=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim = state_dim # 20
        self.action_dim = action_dim # 25
        self.gamma = gamma
        self.batch_size = batch_size

        self.main_network = DDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=lr)

        self.target_network = DDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict()) # same as main_network @init

        self.memory = ReplayMemory(memory_capacity)


    def learn(self, state, action, reward, next_state, done):
        """ Store tuple in memory and do one round of model updates. """
        self.memory.store(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            self.update_model()


    def get_q_values(self, state):
        """ Returns an array with the Q-value associated with each action. """
        q_values = self.main_network(state)
        return q_values.detach().numpy()
            

    def get_action_probs(self, state):
        """ Returns a probability distribution over the action space. """
        q_values = self.main_network(state)
        distribution = F.softmax(q_values, dim=-1)
        return distribution.detach().numpy()


    def select_action(self, state, eval=False):
        """
        Supports both single and batched states. Returns actions as shape (batch_size, 1) or (1, 1).
        Always greedily selects the action with the maximal estimated Q-value.
        """
        q_values = self.main_network(state)
        if eval:
            return torch.argmax(q_values, dim=-1)
        else:
            probs = F.softmax(q_values, dim=-1)
            dist = Categorical(probs=probs)
            return dist.sample()
        

    def update_model(self):
        """
        Update model based on loss between expected Q values (from DDDQN) and actual Q values.
        """
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device)

        q_values = self.main_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_action_values = self.main_network(next_states).max(1)[1].unsqueeze(-1)
        next_q_values = self.target_network(next_states).gather(1, next_action_values).detach().squeeze(-1)

        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = torch.nn.MSELoss()(q_values, expected_q_values) # TODO: try Huber loss?
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

    def hard_update(self):
        """ Periodically update target network parameters to match main network """
        self.target_network.load_state_dict(self.main_network.state_dict())



class CQLAgent(D3QNAgent):
    """
    Essentially D3QN, but performs a Q value update for which the expectation is a lower bound of the actual Q value!
    Should be careful that I don't underestimate too far. Using dueling, but not double, for this reason.
    """

    def cql_loss(self, q_values, current_action):
        """ Computes the CQL loss for a batch of Q-values. Penalizes action values not observed in data. """
        logsumexp = torch.logsumexp(q_values, dim=-1)
        q_a = q_values.gather(-1, current_action)
        return (logsumexp - q_a).mean()
    

    def update_model(self, alpha=0.5):
        """ Update model based on loss between expected Q values (from DDDQN) and actual Q values """
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device) # stochastic

        all_q_values = self.main_network(states)
        cql1_loss = self.cql_loss(all_q_values, actions.unsqueeze(-1))
        q_values = all_q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        next_action_values = self.main_network(next_states).max(1)[1].unsqueeze(-1)
        next_q_values = self.target_network(next_states).gather(1, next_action_values).detach().squeeze(-1)
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        td_loss = torch.nn.MSELoss()(q_values, expected_q_values)
        q1_loss = cql1_loss + alpha * td_loss # TODO tune alpha

        self.optimizer.zero_grad()
        q1_loss.backward()
        self.optimizer.step()



class IQLAgent():

    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, gamma=0.99, temp=3, expectile=0.7, tau=5e-3):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = torch.tensor([gamma])
        self.temp = torch.tensor([temp])
        self.expectile = torch.tensor([expectile])
        self.tau = tau
        
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic1 = Critic(state_dim, action_dim, hidden_dim, 1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim, 1)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(state_dim, action_dim, hidden_dim, 2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim, 2)
        self.critic2_target.load_state_dict(self.critic1.state_dict())

        self.value_net = Value(state_dim, hidden_dim)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)

        
    def select_action(self, state, eval=False):
        """
        During training, sample an action from the logits distribution.
        During evaluation, choose the action with max. logit.
        """
        logits = self.actor(state)

        if eval:
            return torch.argmax(logits, dim=-1)
        else:
            dist = Categorical(logits=logits)
            return dist.sample()
        

    def get_q_values(self, state):
        """  
        Returns a 1D array, containing one Q-value (estimated
        from the target critic network) for each action.
        """
        q_values = []
        for action in range(self.action_dim):
            action = torch.tensor(action)
            with torch.no_grad():
                q1 = self.critic1_target(state, action)
                q2 = self.critic2_target(state, action)
                q_values.append(torch.min(q1, q2))

        return np.array(q_values)
        

    def get_action_probs(self, state):
        """
        Returns a probability distribution over the action space.
        """
        logits = self.actor(state)
        return F.softmax(logits).detach().numpy()
    

    def calc_policy_loss(self, states, actions):
        with torch.no_grad():
            value = self.value_net(states)
            q1 = self.critic1_target(states, actions)
            q2 = self.critic2_target(states, actions)
            min_Q = torch.min(q1, q2) # take the more conservative q-value!

        weights = torch.exp((min_Q - value) * self.temp)
        weights = weights.clamp(max=100.0) # clip high values

        # what is the probability of selecting these observed actions from the actor's learned distribution?
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        actor_loss = -(weights * log_probs).mean()
        return actor_loss
    

    def calc_value_loss(self, states, actions):
        with torch.no_grad():
            q1 = self.critic1_target(states, actions)
            q2 = self.critic2_target(states, actions)
            min_Q = torch.min(q1, q2) # take the more conservative q-value!

        value = self.value_net(states)

        # iql loss
        diff = min_Q - value
        weight = torch.where(diff > 0, self.expectile, (1-self.expectile))
        value_loss = weight * (diff ** 2).mean()
        return value_loss
    

    def calc_q_loss(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_v = self.value_net(next_states)
            q_target = rewards + (self.gamma * (1-dones) * next_v)

        q1 = self.critic1(states, actions) # NOTE: main network
        q2 = self.critic2(states, actions) # NOTE: main network
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        return critic1_loss, critic2_loss
    

    def learn(self, states, actions, rewards, next_states, dones):

        # value loss
        self.value_optimizer.zero_grad()
        value_loss = self.calc_value_loss(states, actions)
        value_loss.backward()
        self.value_optimizer.step()

        # policy loss
        actor_loss = self.calc_policy_loss(states, actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # critic loss (seperately for each network)
        critic1_loss, critic2_loss = self.calc_q_loss(states, actions, rewards, next_states, dones)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        clip_grad_norm_(self.critic1.parameters(), max_norm=1) # TODO is this standard for IQL?
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), max_norm=1)
        self.critic2_optimizer.step()

        # soft update critic target network to match critic network
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

    
    def soft_update(self, local_model, target_model):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)