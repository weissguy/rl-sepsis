import numpy as np
import gymnasium as gym
import pandas as pd
#import torch
from stable_baselines3.common.env_checker import check_env

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lstm_ae import Encoder


class Patient:

    def __init__(self, latent_df, icustayid):
        self.icustayid = icustayid
        self.index = 0
        self.latent_df = latent_df[latent_df['icustayid'] == icustayid]
        self.latent_cols = [f'latent_{num}' for num in range(1, 21)]
        self.mortality = self.latent_df['died_in_hosp'].iloc[0]

    def get_next_state(self):
        if self.is_stay_over():
            raise ValueError('Patient ICU stay is over. No next state available.')
        
        next_state = self.latent_df[self.latent_cols].iloc[self.index].to_numpy(dtype=np.float32)
        self.index += 1
        return next_state

    def is_stay_over(self):
        terminal_state = self.latent_df['terminal_state'].iloc[self.index]
        return terminal_state == 1
    
    def survives(self):
        return self.mortality == 0



class MIMICEnv(gym.Env):

    def __init__(self):

        # continuous, 20-dimensional vector (for now)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        self.obs_dim = 20

        # discrete, (5,5) actions --> maps to 25 discrete 1D actions
        self.action_space = gym.spaces.Discrete(25)
        self.action_dim = 25 # technically (5,5)
        self.actions_to_ids = {(iv, vaso): (5*iv + vaso) for iv in range(5) for vaso in range(5)}
        self.ids_to_actions = {v: k for k, v in self.actions_to_ids.items()}

        # load sepsis_df
        self.latent_df = pd.read_csv('data/latent_states.csv')
        self.icustayids = self.latent_df['icustayid'].unique()

        # load patient data
        self.patient = None

        # load lstm encoder model -- not using here rn bc presaved latent states
        #self.state_encoder = Encoder(input_dim=46, hidden_dim=64, latent_dim=20, dropout=0.0, seq_len=20)
        #self.state_encoder.load_state_dict(torch.load('models/lstm_encoder.pth'))
        #self.state_encoder.eval()

        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.state_encoder.to(self.device)

    
    def reset(self, seed=None, options=None):
        super().reset()

        # some logic for (randomly) choosing a new state
        idx = np.random.choice(self.icustayids)
        self.patient = Patient(self.latent_df, idx)

        observation = self._get_obs(done=False)
        info = {}

        return observation, info
    
    def _get_obs(self, done):
        """
        Returns the current state of the patient (as a latent vector).
        If the patient is in a terminal state, returns an array of zeros.
        """
        if done:
            return np.zeros(self.obs_dim) # dummy state
        else:
            return self.patient.get_next_state()
    
    def _get_reward(self, done):
        """
        Assumes that the current state is a terminal state. 
        +1 if patient survives, -1 if they die.
        """
        if done and self.patient.survives():
            return 15
        elif done and not self.patient.survives():
            return -15
        else:
            return 0
        
    
    def step(self, action):

        # check if the episode is done
        done = self.patient.is_stay_over()

        new_state = self._get_obs(done)
        reward = self._get_reward(done)

        return new_state, reward, done, False, {}



if __name__ == '__main__':
    env = MIMICEnv()
    check_env(env)
