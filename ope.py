import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.adam
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from common import *
from rl_policies import *
from networks import BehaviorPolicy, FQENet
from envs import MIMICEnv



class OfflineDataset(Dataset):
    """
    Interface that allows us to access a series of (s_t, a, r, s_{t+1}) 
    tuples from our offline RL gym environment. This lets us do batched 
    training using the PyTorch TensorDataset and DataLoader interfaces.

    Takes in a df --> allows us to make separate train/test/val datasets.
    """

    def __init__(self, df, latent):
        """
        Step through the environment, collecting observations as we go.
        """
        env = MIMICEnv(df, latent)
        self.samples = []
        # collect data for each unique icu stay / trajectory
        for _ in range(len(df['icustayid'].unique())):
            state = env.reset(stepwise=True)
            done = False
            while not done:
                action = env.get_logged_action()
                next_state, reward, done, _ = env.step(action)
                self.samples.append((state, action, reward, next_state))
                state = next_state

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s, a, r, s_next = self.samples[idx]
        return (torch.tensor(s).float(),
                torch.tensor(a).long(),
                torch.tensor(r).float(),
                torch.tensor(s_next).float())



def compute_physician_policy(train_df, batch_size=64, verbose=True):
    """
    Learn a function over the action space that mimics the physician's actions.
    Rather than just knowing the single action the physician took, we can have
    a probability distribution of actions they might have taken.
    """

    loader = DataLoader(OfflineDataset(train_df, latent=False), batch_size=batch_size, shuffle=True)

    model = BehaviorPolicy(state_dim=46, action_dim=25, hidden_dim=256)
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

    loader = DataLoader(OfflineDataset(test_df, latent=False), shuffle=True) # batch size = 1

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


    

def train_fqe(train_df, agent_policy, latent=True, batch_size=64, lr=0.001, gamma=0.99):

    loader = DataLoader(OfflineDataset(train_df, latent), batch_size=batch_size) # don't shuffle, I think

    fqe_net = FQENet(state_dim=20, action_dim=25, hidden_dim=64)
    optimizer = torch.optim.Adam(fqe_net.parameters(), lr=lr)

    n_epochs = 5
    for _ in range(n_epochs):
        for state, action, reward, next_state in tqdm(loader):

            q_vals = fqe_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

            next_action = agent_policy.select_action(state)
            next_q_vals = fqe_net(next_state).gather(1, next_action.unsqueeze(1)).squeeze(1)
            target_q_vals = reward + (gamma * next_q_vals)

            loss = nn.MSELoss()(q_vals, target_q_vals)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return fqe_net



def ope_fqe(train_df, sepsis_df, agent_policy, return_fqe_net=False):
    """
    Fitted Q-Evaluation. Estimates the Q-function for agent_policy using fqe_net, by
    sampling from trajectories in the offline dataset.

    Assumes that fqe_net has already been trained. Returns Q_hat(s_0, pi_0).
    """

    fqe_net = train_fqe(train_df, agent_policy)

    init_states = get_states(sepsis_df[sepsis_df['initial_state'] == 1])

    with torch.no_grad():
        init_actions = agent_policy.select_action(init_states)
        q_values = fqe_net(init_states).gather(1, init_actions.unsqueeze(1)).squeeze(1)
        mean_q_val = q_values.mean().item()
    
    if return_fqe_net:
        return mean_q_val, fqe_net
    else:
        return mean_q_val



def ope_wis(sepsis_df, physician_policy, agent_policy, gamma=0.99):
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

    def compute_rho(trajectory, agent_policy, physician_policy):
        """
        Helper function. Computes the cumulative product of the agent/physician ratios
        for the selection action, for a single trajectory.
        """

        states = get_states(trajectory)
        actions = trajectory.apply(get_action_id, axis=1).values

        agent_probs = np.array([agent_policy.get_action_probs(state) for state in states]) # (x,25)
        agent_action_probs = agent_probs[np.arange(len(actions)), actions] # prob of action that was taken (from offline data)

        physician_probs = np.array([physician_policy.get_action_probs(state) for state in states]) # (x,25)
        physician_action_probs = physician_probs[np.arange(len(actions)), actions] # prob of action that was taken (from offline data)

        eps = 1e-8 # avoid div. by zero
        ratios = agent_action_probs / (physician_action_probs + eps)
        ratios = np.clip(ratios, a_min=1e-3, a_max=100.0) # clip values to prevent "explosions"

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
    rho_array = np.array([compute_rho(trajectory, agent_policy, physician_policy) for trajectory in trajectories])
    rho_array = rho_array / np.nansum(rho_array) # normalize for WIS

    def compute_trial_estimate(trajectory):
        rewards = trajectory['reward']
        discounts = gamma ** np.arange(len(rewards))
        return np.sum(discounts * rewards)

    # list containing one estimated V value per trajectory
    individual_trial_estimates = [compute_trial_estimate(trajectory) for trajectory in trajectories]
    estimate = np.nansum(individual_trial_estimates * rho_array)

    return estimate



def ope_doubly_robust(train_df, sepsis_df, physician_policy, agent_policy, gamma=0.99):
    """
    Off-policy evaluation with the Doubly Robust method (DR). Given a set of trajectories,
    a behavioral policy (from compute_physician_policy), and a policy to evaluate (from model),
    we return an estimate of how well our policy performs on offline data.

    We do this by combining WIS and FQE.
    """

    fqe_estimate, fqe_net = ope_fqe(train_df, sepsis_df, agent_policy, return_fqe_net=True)
    wis_estimate = ope_wis(sepsis_df, physician_policy, agent_policy, gamma)

    icustayids = sepsis_df['icustayid'].unique()
    trajectories = [sepsis_df[sepsis_df['icustayid'] == idx] for idx in icustayids] # list of dfs

    # correction
    correction_total = 0.0
    for trajectory in trajectories:
        states = get_states(trajectory)
        actions = trajectory.apply(get_action_id, axis=1).values
        rewards = trajectory['reward'].values
        discounts = gamma ** np.arange(len(rewards))

        # get action probabilities for importance weights
        agent_probs = np.array([agent_policy.get_action_probs(state) for state in states])
        agent_action_probs = agent_probs[np.arange(len(actions)), actions]
        physician_probs = np.array([physician_policy.get_action_probs(state) for state in states])
        physician_action_probs = physician_probs[np.arange(len(actions)), actions]
        eps = 1e-8
        ratios = agent_action_probs / (physician_action_probs + eps)
        cumprod_ratios = np.cumprod(ratios)

        with torch.no_grad():
            # q-values for behavioral policy actions
            q_values = fqe_net(states)
            q_taken = q_values[np.arange(len(actions)), actions].numpy()

            # q-values for evaluation policy actions
            agent_actions = agent_policy.select_action(states).squeeze(-1).numpy()
            q_agent = q_values[np.arange(len(actions)), agent_actions].numpy()

        # DR correction for this trajectory
        correction = np.sum(discounts * cumprod_ratios * (rewards - q_taken + q_agent))
        correction_total += correction

    # normalize correction by number of trajectories (or sum of weights, depending on convention)
    correction_term = correction_total / len(trajectories)
    

    return wis_estimate + fqe_estimate - correction_term
    


