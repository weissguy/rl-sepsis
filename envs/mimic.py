import numpy as np
import gymnasium as gym
import pandas as pd
import torch
from stable_baselines3.common.env_checker import check_env

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lstm_ae import Encoder


class Patient:

    def __init__(self, patient_df, latent_df, icustayid):
        self.patient_df = patient_df.loc[patient_df['icustayid'] == icustayid]
        self.latent_df = latent_df.loc[latent_df['icustayid'] == icustayid]
        self.mortality = self.patient_df['died_in_hosp'].values[0]

    def get_patient_data(self, index):
        if self.is_stay_over(index):
            raise ValueError('Patient ICU stay is over. No next state available.')
    
        return self.latent_df.iloc[index].to_numpy(dtype=np.float32)

    def is_stay_over(self, index):
        return index >= len(self.latent_df)
    
    def survives(self):
        return self.mortality == 0



class MIMICEnv(gym.Env):

    def __init__(self):

        # continuous, 20-dimensional vector (for now)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        self.obs_dim = (20,)

        # discrete, (5,5) actions
        self.action_space = gym.spaces.MultiDiscrete([5, 5])
        self.action_dim = (5, 5)

        # load sepsis_df
        self.sepsis_df = pd.read_csv('data/sepsis_df.csv')
        self.icustayids = self.sepsis_df['icustayid'].unique()
        self.latent_df = pd.read_csv('data/latent_states.csv')

        # load patient data
        self.patient_id = None
        self.patient = None
        self.current_index = None

        # load lstm encoder model
        self.state_encoder = Encoder(input_dim=44, hidden_dim=256, latent_dim=20, dropout=0.0, seq_len=50)
        self.state_encoder.load_state_dict(torch.load('models/lstm_encoder.pth'))
        self.state_encoder.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_encoder.to(self.device)

    
    def reset(self, seed=None, options=None):
        super().reset()

        # some logic for (randomly?) choosing a new state
        self.patient_id = np.random.choice(self.icustayids)
        self.patient = Patient(self.sepsis_df, self.latent_df, self.patient_id)
        self.current_index = 0

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
            state_data = self.patient.get_patient_data(self.current_index)
            # get the vector from the pre-saved csv
            return state_data
    
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

        self.current_index += 1

        # check if the episode is done
        done = self.patient.is_stay_over(self.current_index)

        updated_state = self._get_obs(done)
        reward = self._get_reward(done)

        return updated_state, reward, done, False, {}



if __name__ == '__main__':
    env = MIMICEnv()
    check_env(env)
