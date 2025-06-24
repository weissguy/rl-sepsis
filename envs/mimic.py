import numpy as np
import gymnasium as gym
import pandas as pd
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lstm_ae import Encoder


class Patient:

    def __init__(self, df, icustayid):
        self.patient_df = df.loc[df['icustayid'] == icustayid]
        self.mortality = self.patient_df['died_in_hosp_or_within_48h'].values[0]

    def get_state(self, index):
        if self.is_stay_over(index):
            raise ValueError('Patient ICU stay is over. No next state available.')
        
        state_data = self.patient_df.iloc[index]
        state_tensor = torch.tensor(state_data.values, dtype=torch.float32).unsqueeze(0) # TODO unsqueeze?
        with torch.no_grad():
            latent = self.state_encoder(state_tensor)

        return latent

    def is_stay_over(self, index):
        return index >= len(self.patient_df)
    
    def survives(self):
        return self.mortality == 0



class MIMICEnv(gym.Env):

    def __init__(self):

        # continuous, 44-dimensional vector (for now)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32)
        self.obs_dim = (44,)

        # discrete, (5,5) actions
        self.action_space = gym.spaces.MultiDiscrete([5, 5])
        self.action_dim = (5, 5)
        self.action_log = {}

        # load sepsis_df
        self.sepsis_df = pd.read_csv('data/MIMICtable.csv') # TODO: change to sepsis_df
        self.icustayids = self.sepsis_df['icustayid'].unique()

        # load patient data
        self.patient_id = None
        self.patient = None
        self.current_index = None

        # load lstm encoder model
        self.state_encoder = Encoder(input_dim=44, hidden_dim=256, latent_dim=20, dropout=0.0, seq_len=50)
        self.state_encoder.load_state_dict(torch.load('models/lstm_encoder.pth'))
        self.state_encoder.eval()

    
    def reset(self, seed=None, options=None):
        super().reset(seed, options)

        # some logic for (randomly?) choosing a new state
        self.patient_id = np.random.choice(self.icustayids)
        self.patient = Patient(self.sepsis_df, self.patient_id)
        self.current_index = 0

        self.action_log[self.patient_id] = []

        observation = self.patient.get_next_state()
        info = {}

        return observation, info
    
    def _get_obs(self, done):
        """
        Returns the current state of the patient.
        If the patient is in a terminal state, returns an array of zeros.
        """
        if done:
            return np.zeros(self.state_encoder.latent_dim) # dummy state
        else:
            return self.patient.get_state(self.current_index)
    
    def _get_reward(self, done):
        """
        Assumes that the current state is a terminal state. 
        +1 if patient survives, -1 if they die.
        """
        if done:
            if self.patient.survives():
                return 15
            else:
                return -15
        else:
            return 0
    

    def step(self, action):

        self.current_index += 1

        # log the action
        # TODO: action is a np array
        self.action_log[self.patient_id].append(action)

        # check if the episode is done
        done = self.patient.is_stay_over(self.current_index)

        updated_state = self._get_obs(done)
        reward = self._get_reward(done)

        return updated_state, reward, done, False, {}



env = MIMICEnv()