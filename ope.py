### Off-Policy Evaluation (OPE) ###

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from q_networks import DDQNAgent



class BehaviorPolicyNet(nn.Module):
    """
    Learn a behavior policy that mimics clinician actions taken in the data. 

    Input: a latent state vector (20,)
    Output: a probability distribution over the discrete action space (25,)
    """

    def __init__(self, state_dim=20, hidden_dim=64, action_dim=25):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x.cpu().data.numpy()
        
    def get_action_probs(self, state):
        """ Returns a probability distribution over the actions. """
        logits = self.forward(state)
        return F.softmax(logits)
    


def compute_physician_policy(physician_df, batch_size=32):
    """
    Learn a function over the action space that mimics the physician's actions.
    Rather than just knowing the single action the physician took, we can have
    a probability distribution of actions they might have taken.
    """

    states = physician_df['latent_state'] # series of strings --> tensor of arrays
    print(states[1])
    states = torch.tensor(states, dtype=torch.float32)

    def get_action_id(row):
        return (5 * row['iv_bin']) + row['vaso_bin']
    
    actions = physician_df.apply(get_action_id, axis=1)
    actions = torch.tensor(actions, dtype=torch.long)

    # train behavior policy net
    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # shuffle, ignoring trajectories

    model = BehaviorPolicyNet(state_dim=states.shape[0], hidden_dim=128, action_dim=actions.shape[0])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    N_EPOCHS = 10
    for _ in range(N_EPOCHS):
        for xb, yb in loader:
            logits = model(xb, logits=True)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model



def off_policy_eval_wis(physician_df, physician_policy, agent_policy, gamma=0.99):
    """
    Off-policy evaluation with Weighted Importance Sampling (WIS). Given a set of trajectories,
    a behavioral policy (from compute_physician_policy), and a policy to evaluate (from model),
    we return an estimate of how well our policy performs on offline data.

    We do this by considering the cumulative importance weight for a trajectory, which is the
    product over all timesteps of the probability under the evaluation policy divided by the
    probability under the behavioral policy.

    Note: might suffer from high variance.
    """

    icustayids = physician_df['icustayid'].unique()
    trajectories = [physician_df[physician_df['icustayid'] == idx] for idx in icustayids] # list of dfs

    def compute_rho(trajectory):
        agent_probs = [agent_policy.get_action_probs(state) for state in trajectory['latent_state']] # np array (25,)
        physician_probs = [physician_policy.get_action_probs(state) for state in trajectory['latent_state']] # TODO: ermmm
        return np.prod(agent_probs / physician_probs)

    # list containing one cumulative row per trajectory
    rho_array = [compute_rho(trajectory) for trajectory in trajectories]
    rho_array = rho_array / np.nansum(rho_array) # normalize

    def compute_trial_estimate(trajectory):
        rewards = trajectory['reward']
        discounts = gamma ** np.arange(len(rewards))
        return np.sum(discounts * rewards)

    # list containing one estimated V value per trajectory
    individual_trial_estimates = [compute_trial_estimate(trajectory) for trajectory in trajectories]
    estimate = np.nansum(individual_trial_estimates * rho_array)

    return estimate



def off_policy_eval_dr(physician_df, model, gamma=0.99):
    """
    Off-policy evaluation with the Doubly Robust method (DR). Given a set of trajectories,
    a behavioral policy (from compute_physician_policy), and a policy to evaluate (from model),
    we return an estimate of how well our policy performs on offline data.

    We do this by combining WIS with FQE
    """
    pass



    


if __name__ == '__main__':
    sepsis_df = pd.read_csv('data/sepsis_df.csv') # TODO: full df or just test?

    physician_policy = compute_physician_policy(sepsis_df)
    #model = DDQNAgent(state_dim=20, action_dim=20)
    #off_policy_eval_wis(sepsis_df, physician_policy, model)
