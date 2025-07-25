import torch
import torch.nn as nn
import torch.nn.functional as F


class DDQN(nn.Module):
    """ Dueling Deep Q-Network """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.advantage = torch.nn.Linear(hidden_dim, action_dim)
        self.value = torch.nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """
        Forward pass. Two linear layers. Splits the output into separate layers for
        the value (one value for the current state) and advantage (one value per action).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        advantage = self.advantage(x)
        value = self.value(x)

        return value + (advantage - advantage.mean()) # (25,)
    


class Actor(nn.Module):
    """
    Actor (Policy) Model. Computes pi(s|a).
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.out(x)
    

    
class Critic(nn.Module):
    """
    Critic (Value) Model. Computes an estimate for Q(s,a).
    """

    def __init__(self, state_dim, action_dim, hidden_dim, seed):
        super().__init__()
        torch.manual_seed(seed)
        # returns 1 value per state/action pair (1D action in range [0,24])
        self.fc1 = nn.Linear(state_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        

    def forward(self, state, action):
        x = torch.cat([state, action.unsqueeze(-1)], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
    

class Value(nn.Module):
    """
    Value (Value) Model. Computes an estimate for V(s).
    """

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        # returns 1 value per state
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.out(x)
    

class BehaviorPolicy(nn.Module):
    """
    Learn a behavior policy that mimics clinician actions taken in the data. 

    Input: a latent state vector (20,)
    Output: the predicted probability of choosing each action (25,)
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """ Returns logits over the action space. """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)

        return x
        
    def get_action_probs(self, state):
        """ Returns a probability distribution over the action space. """
        logits = self.forward(state)
        distribution = F.softmax(logits)
        return distribution.detach().numpy()
    
    def select_action(self, state):
        """
        Supports both single and batched states. Returns actions as shape (batch_size, 1) or (1, 1).
        """
        logits = self.forward(state)
        return torch.argmax(logits, dim=-1)
    


class FQENet(nn.Module):
    """
    Learn a Q-function corresponding to a policy, for off-policy evaluation.

    Input: a latent state vector (20,)
    Output: a Q-function over the action space (25,)
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.out(x)