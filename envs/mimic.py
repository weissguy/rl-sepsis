import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ICUStay:
    """
    Stores the information for a single trajectory, corresponding to one ICU stay ID.
    """

    def __init__(self, sepsis_df, icustayid):
        self.index = 0

        df = sepsis_df[sepsis_df['icustayid'] == icustayid]
        self.df = df # TODO remove
        self.icustayid = icustayid # TODO remove
        self.mortality = df['died_in_hosp'].iloc[0]

        # each state is a string rep of a np array --> convert to np array
        self.states = df['latent_state']
        self.states = np.stack([np.fromstring(state.strip('[]'), sep=' ', dtype=np.float32) for state in self.states])

        # convert each (iv, vaso) action into a 1D action id
        self.actions = df.apply(lambda row: (5 * row['iv_bin']) + row['vaso_bin'], axis=1).reset_index(drop=True)


    def increment_index(self):
        self.index += 1

    def get_state(self):
        if self.is_stay_over():
            raise ValueError('Patient ICU stay is over. No state available.')
        
        return self.states[self.index]
    
    def get_action(self):
        if self.is_stay_over():
            raise ValueError('Patient ICU stay is over. No logged action available.')
        
        return self.actions[self.index]

    def is_stay_over(self):
        return self.index >= len(self.states)
    
    def survives(self):
        return self.mortality == 0



class MIMICEnv(gym.Env):

    def __init__(self, df):

        # continuous, 20-dimensional vector (for now)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        self.obs_dim = 20

        # discrete, (5,5) actions --> maps to 25 discrete 1D actions
        self.action_space = gym.spaces.Discrete(25)
        self.action_dim = 25

        # load sepsis_df
        self.sepsis_df = df
        self.icustayids = self.sepsis_df['icustayid'].unique()

        # initialize data for one stay
        self.current_id = None
        self.icustay = None

    
    def reset(self, stepwise=False):
        super().reset()

        if stepwise:
            if self.current_id is None:
                # initalize to first id
                new_id = self.icustayids[0]
            else:
                # step forward to the next id in the sequence
                old_id = self.current_id
                new_index = np.where(self.icustayids == old_id)[0][0] + 1
                if new_index >= len(self.icustayids):
                    raise StopIteration('reached the end of icustayids')
                new_id = self.icustayids[new_index]
        else:
            # randomly choose a new trajectory
            new_id = np.random.choice(self.icustayids)

        self.current_id = new_id
        self.icustay = ICUStay(self.sepsis_df, new_id)

        observation = self._get_obs(done=False)
        info = {}

        return observation, info
    
    def _get_obs(self, done):
        """
        Returns the current state of the icu stay (as a latent vector).
        If the icu stay is in a terminal state, returns an array of zeros.
        """
        if done:
            return np.zeros(self.obs_dim, dtype=np.float32) # dummy state
        else:
            return self.icustay.get_state()
    
    def _get_reward(self, done):
        """
        Assumes that the current state is a terminal state. 
        +1 if patient survives, -1 if they die.
        """
        if done and self.icustay.survives():
            return 15
        elif done and not self.icustay.survives():
            return -15
        else:
            return 0
        
    
    def get_logged_action(self):
        """
        Returns the 1D id for the action logged by the physician at the
        current timestep.
        """
        return self.icustay.get_action()
        
    
    def step(self, action):

        # step forward one timestep
        self.icustay.increment_index()

        # check if this is the last timestep
        done = self.icustay.is_stay_over()
        new_state = self._get_obs(done)
        reward = self._get_reward(done)

        return new_state, reward, done, False, {}



if __name__ == '__main__':
    env = MIMICEnv()
    check_env(env)
