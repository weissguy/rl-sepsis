### Off-Policy Evaluation (OPE) ###

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import zscore
import matplotlib.pyplot as plt
from tqdm import tqdm

from q_networks import DDQNAgent



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
        return F.softmax(logits)
    


def compute_physician_policy(train_df, batch_size=32, verbose=True):
    """
    Learn a function over the action space that mimics the physician's actions.
    Rather than just knowing the single action the physician took, we can have
    a probability distribution of actions they might have taken.
    """

    states = train_df['latent_state']
    # each state is a string rep of a np array --> convert to np array
    states = np.stack([np.fromstring(state.strip('[]'), sep=' ') for state in states])
    states = torch.from_numpy(states) # (264589, 20)
    states = states.to(dtype=torch.float32)

    def get_action_id(row):
        """
        Given an IV bin and a vaso bin, return the unique action ID, in [0, 24].
        """
        return (5 * row['iv_bin']) + row['vaso_bin']
    
    # for each timestep, form a probability distribution over the action space
    actions = train_df.apply(get_action_id, axis=1)
    actions = torch.tensor(actions.values, dtype=torch.long)

    # train behavior policy network using physician actions
    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # shuffle, ignoring trajectories

    model = BehaviorPolicyNet()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    N_EPOCHS = 5
    losses = []
    for _ in range(N_EPOCHS):
        for x_batch, y_batch in tqdm(loader):
            logits = model(x_batch)
            loss = F.cross_entropy(logits, y_batch)
            if _ == 0:
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

    states = test_df['latent_state']
    # each state is a string rep of a np array --> convert to np array
    states = np.stack([np.fromstring(state.strip('[]'), sep=' ') for state in states])
    states = torch.from_numpy(states) # (264589, 20)
    states = states.to(dtype=torch.float32)

    def get_action_id(row):
        """
        Given an IV bin and a vaso bin, return the unique action ID, in [0, 24].
        """
        return (5 * row['iv_bin']) + row['vaso_bin']
    
    # for each timestep, form a probability distribution over the action space
    actions = test_df.apply(get_action_id, axis=1)
    actions = torch.tensor(actions.values, dtype=torch.long)

    # evaluate behavior policy network using physician actions
    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, shuffle=True) # shuffle, ignoring trajectories

    losses, all_probs = [], []
    physician_policy.eval()
    for x_batch, y_batch in tqdm(loader):
        logits = physician_policy(x_batch)
        loss = F.cross_entropy(logits, y_batch)
        losses.append(loss.item())
        probs = F.softmax(logits, dim=-1).detach().numpy()
        all_probs.append(probs)

    # plot losses
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title('Evaluation Losses')
    plt.show()
    print(f'mean eval loss: {np.mean(losses)}')

    # predicted-action histogram
    all_probs = np.concatenate(all_probs, axis=0)
    avg_probs = np.mean(all_probs, axis=0)
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(avg_probs)), avg_probs)
    ax.set_xlabel("Action")
    ax.set_ylabel("Averaged Predicted Probability")
    plt.show()

    # TODO: visualize predicted action in PCA of state space



def off_policy_eval_wis(physician_df, physician_policy, agent_policy, gamma=0.99):
    """
    Off-policy evaluation with Weighted Importance Sampling (WIS). Given a set of trajectories,
    a behavioral policy (from compute_physician_policy), and a policy to evaluate (from model),
    we return an estimate of how well our policy performs on offline data (the whole dataset).

    We do this by considering the cumulative importance weight for a trajectory, which is the
    product over all timesteps of the probability under the evaluation policy divided by the
    probability under the behavioral policy.

    Note: might suffer from high variance.
    """

    icustayids = physician_df['icustayid'].unique()
    trajectories = [physician_df[physician_df['icustayid'] == idx] for idx in icustayids] # list of dfs

    def compute_rho(trajectory):
        agent_probs = [agent_policy.get_action_probs(state) for state in trajectory['latent_state']] # np array (25,)
        physician_probs = [physician_policy.get_action_probs(state) for state in trajectory['latent_state']] # np array (25,)
        return np.prod(agent_probs / physician_probs) # float?

    # list containing one cumulative rho per trajectory
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

    sepsis_df = pd.read_csv('data/sepsis_df.csv')
    sepsis_df = sepsis_df.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle

    cutoff = int(0.8 * len(sepsis_df))
    train_df = sepsis_df[:cutoff]
    test_df = sepsis_df[cutoff:]

    # train
    #physician_policy = compute_physician_policy(train_df, verbose=True)
    #torch.save(physician_policy.state_dict(), 'models/physician_policy.pth')


    physician_policy = BehaviorPolicyNet()
    physician_policy.load_state_dict(torch.load('models/physician_policy.pth'))

    # evaluate
    evaluate_physician_policy(physician_policy, test_df)

    #model = DDQNAgent(state_dim=20, action_dim=20)
    #off_policy_eval_wis(sepsis_df, physician_policy, model)
