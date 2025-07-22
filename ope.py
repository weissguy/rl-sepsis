### Off-Policy Evaluation (OPE) ###

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.adam
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

from rl_policies import *
from envs.mimic import MIMICEnv



class OfflineDataset(torch.utils.data.Dataset):
    """
    Interface that allows us to access a series of (s_t, a, r, s_{t+1}) 
    tuples from our offline RL gym environment. This lets us do batched 
    training using the PyTorch TensorDataset and DataLoader interfaces.
    """

    def __init__(self, df):
        """
        Step through the environment, collecting observations as we go.
        """
        env = MIMICEnv(df)
        self.samples = []
        # collect data for each unique icu stay / trajectory
        for _ in range(len(df['icustayid'].unique())):
            state, _ = env.reset(stepwise=True)
            done = False
            while not done:
                action = env.get_logged_action()
                next_state, reward, terminated, truncated, _ = env.step(action)
                self.samples.append((state, action, reward, next_state))
                done = terminated or truncated
                state = next_state

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s, a, r, s_next = self.samples[idx]
        return (torch.tensor(s).float(),
                torch.tensor(a).long(),
                torch.tensor(r).float(),
                torch.tensor(s_next).float())




class BehaviorPolicyNet(nn.Module):
    """
    Learn a behavior policy that mimics clinician actions taken in the data. 

    Input: a latent state vector (20,)
    Output: the predicted probability of choosing each action (25,)
    """

    def __init__(self, state_dim=20, hidden_dim=256, action_dim=25):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """ Returns logits over the action space. """
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
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
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Convert to batch of 1
        logits = self.forward(state)
        return torch.argmax(logits, dim=1, keepdim=True)
    
    

def get_states(df):
    """
    Helper function. Given a df, returns a Tensor of all latent space representations 
    in that df. Pandas DataFrames store the string representation of arrays, so we have to 
    parse the contents of each cell.
    """
    states = df['latent_state']
    states = np.stack([np.fromstring(state.strip('[]'), sep=' ') for state in states])
    states = torch.from_numpy(states).to(dtype=torch.float32) # (264589, 20)
    return states


def get_action_id(row):
    """
    Helper function. Given an IV bin and a vaso bin, return the unique action ID, in [0, 24].
    """
    return (5 * row['iv_bin']) + row['vaso_bin']


def compute_physician_policy(train_df, batch_size=64, verbose=True):
    """
    Learn a function over the action space that mimics the physician's actions.
    Rather than just knowing the single action the physician took, we can have
    a probability distribution of actions they might have taken.
    """

    loader = DataLoader(OfflineDataset(train_df), batch_size=batch_size, shuffle=True)

    model = BehaviorPolicyNet()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 5
    losses = []
    for epoch in range(n_epochs):
        for states, actions, _, _ in tqdm(loader):
            logits = model(states)
            loss = F.cross_entropy(logits, actions)
            if epoch == 0:
                losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if verbose:
        # plot losses
        fig, ax = plt.subplots()
        ax.plot(losses)
        ax.set_title('Losses during training')
        plt.show()

        print(f'mean training loss: {np.mean(losses)}')

    return model



def evaluate_physician_policy(physician_policy, test_df):

    loader = DataLoader(OfflineDataset(test_df), shuffle=True) # batch size = 1

    losses = []
    action_hist = [0] * 25
    physician_policy.eval()
    for state, action, _, _ in tqdm(loader):
        logits = physician_policy(state)
        loss = F.cross_entropy(logits, action)
        losses.append(loss.item())
        action_hist[logits.argmax()] += 1

    # plot losses
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title('Evaluation Losses')
    plt.show()
    print(f'mean eval loss: {np.mean(losses)}')

    # predicted-action histogram
    fig, ax = plt.subplots()
    ax.plot(action_hist)
    ax.set_xlabel("Action")
    ax.set_ylabel("Frequency")
    plt.show()



class FQENet(nn.Module):
    """
    Learn a Q-function corresponding to a policy, for off-policy evaluation.

    Input: a latent state vector (20,)
    Output: a Q-function over the action space (25,)
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.out(x)
    

def train_fqe(train_df, agent_policy, batch_size=64, lr=0.001, gamma=0.99):

    loader = DataLoader(OfflineDataset(train_df), batch_size=batch_size) # don't shuffle, I think

    fqe_net = FQENet(state_dim=20, action_dim=25)
    optimizer = torch.optim.Adam(fqe_net.parameters(), lr=lr)

    n_epochs = 5
    for _ in range(n_epochs):
        for state, action, reward, next_state in tqdm(loader):

            q_vals = fqe_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

            next_action = agent_policy.select_action(state)
            next_q_vals = fqe_net(next_state).gather(1, next_action).squeeze(1)
            target_q_vals = reward + (gamma * next_q_vals)

            loss = nn.MSELoss()(q_vals, target_q_vals)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return fqe_net



def off_policy_eval_fqe(train_df, sepsis_df, agent_policy):
    """
    Fitted Q-Evaluation. Estimates the Q-function for agent_policy using fqe_net, by
    sampling from trajectories in the offline dataset.

    Assumes that fqe_net has already been trained. Returns Q_hat(s_0, pi_0).
    """

    fqe_net = train_fqe(train_df, agent_policy)

    init_states = get_states(sepsis_df[sepsis_df['initial_state'] == 1])

    with torch.no_grad():
        init_actions = agent_policy.select_action(init_states)
        q_values = fqe_net(init_states).gather(1, init_actions)
        return q_values.mean().item()



def off_policy_eval_wis(sepsis_df, physician_policy, agent_policy, gamma=0.99):
    """
    Off-policy evaluation with Weighted Importance Sampling (WIS). Given a set of trajectories,
    a behavioral policy (from compute_physician_policy), and a policy to evaluate (from model),
    we return an estimate of how well our policy performs on offline data (the whole dataset).

    We do this by considering the cumulative importance weight for a trajectory, which is the
    product over all timesteps of the probability of choosing the observed action under the evaluation 
    policy, divided by the probability of choosing the observed action under the behavioral policy.

    Note: might suffer from high variance.
    """

    icustayids = sepsis_df['icustayid'].unique()
    trajectories = [sepsis_df[sepsis_df['icustayid'] == idx] for idx in icustayids] # list of dfs

    def compute_rho(trajectory):

        states = get_states(trajectory)
        actions = trajectory.apply(get_action_id, axis=1).values

        agent_probs = np.array([agent_policy.get_action_probs(state) for state in states]) # (x,25)
        agent_action_probs = agent_probs[np.arange(len(actions)), actions] # prob of action that was taken

        physician_probs = np.array([physician_policy.get_action_probs(state) for state in states]) # (x,25)
        physician_action_probs = physician_probs[np.arange(len(actions)), actions] # prob of action that was taken

        eps = 1e-8 # avoid div. by zero
        ratios = agent_action_probs / (physician_action_probs + eps)
        return np.prod(ratios)

        """
        # visualize agent and physician policies
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(agent_probs[0])
        ax1.set_title('agent probability distribution')
        
        ax2.plot(physician_probs[0])
        ax2.set_title('physician probability distribution')
        plt.show()
        """

    # list containing one cumulative rho per trajectory
    rho_array = np.array([compute_rho(trajectory) for trajectory in trajectories])
    # normalize for WIS
    rho_array = rho_array / np.nansum(rho_array)

    # TODO: clip rhos?

    def compute_trial_estimate(trajectory):
        rewards = trajectory['reward']
        discounts = gamma ** np.arange(len(rewards))
        return np.sum(discounts * rewards)

    # list containing one estimated V value per trajectory
    individual_trial_estimates = [compute_trial_estimate(trajectory) for trajectory in trajectories]
    estimate = np.nansum(individual_trial_estimates * rho_array)

    return estimate



def off_policy_eval_dr(sepsis_df, train_df, physician_policy, agent_policy, gamma=0.99):
    """
    Off-policy evaluation with the Doubly Robust method (DR). Given a set of trajectories,
    a behavioral policy (from compute_physician_policy), and a policy to evaluate (from model),
    we return an estimate of how well our policy performs on offline data.

    We do this by combining WIS and FQE.
    """

    ### WIS ###

    icustayids = sepsis_df['icustayid'].unique()
    trajectories = [sepsis_df[sepsis_df['icustayid'] == idx] for idx in icustayids] # list of dfs

    def compute_rho(trajectory):

        states = get_states(trajectory)
        actions = trajectory.apply(get_action_id, axis=1).values

        agent_probs = np.array([agent_policy.get_action_probs(state) for state in states]) # (x,25)
        agent_action_probs = agent_probs[np.arange(len(actions)), actions] # prob of action that was taken

        physician_probs = np.array([physician_policy.get_action_probs(state) for state in states]) # (x,25)
        physician_action_probs = physician_probs[np.arange(len(actions)), actions] # prob of action that was taken

        eps = 1e-8 # avoid div. by zero
        ratios = agent_action_probs / (physician_action_probs + eps)
        return np.prod(ratios)


    # list containing one cumulative rho per trajectory
    rho_array = np.array([compute_rho(trajectory) for trajectory in trajectories])
    # normalize for WIS
    rho_array = rho_array / np.nansum(rho_array)

    # TODO: clip rhos?

    def compute_trial_estimate(trajectory):
        rewards = trajectory['reward']
        discounts = gamma ** np.arange(len(rewards))
        return np.sum(discounts * rewards)

    # list containing one estimated V value per trajectory
    individual_trial_estimates = [compute_trial_estimate(trajectory) for trajectory in trajectories]
    wis_estimate = np.nansum(individual_trial_estimates * rho_array)

    
    ### FQE ###

    fqe_net = train_fqe(train_df, agent_policy)

    init_states = get_states(sepsis_df[sepsis_df['initial_state'] == 1])

    with torch.no_grad():
        init_actions = agent_policy.select_action(init_states)
        q_values = fqe_net(init_states).gather(1, init_actions)
        fqe_estimate = q_values.mean().item()


    # correction

    correction_term = rho_array * 



    return wis_estimate + fqe_estimate - correction_term
    


