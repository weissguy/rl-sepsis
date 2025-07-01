import gymnasium as gym
from envs.mimic import MIMICEnv
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import matplotlib.pyplot as plt


env = gym.make('mimic-v0')


def eval_policy(model, env, verbose=True):

    returns, lengths = evaluate_policy(model, env, n_eval_episodes=20, return_episode_rewards=True)

    if verbose:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title('return / episode')
        ax1.plot(returns)
        ax2.set_title('episode length')
        ax2.plot(lengths)

        plt.show()

    return np.mean(returns), np.mean(lengths)


### Proximal Policy Optimization ###

#ppo_model = PPO('MlpPolicy', env, verbose=1).learn(total_timesteps=1000)
#ppo_model.save('models/mimic_ppo')

#ppo_model = PPO.load('models/mimic_ppo')

#ppo_return, ppo_len = eval_policy(ppo_model, env)



### Deep Q Networks ###

dqn_model = DQN('MlpPolicy', env, verbose=1).learn(total_timesteps=1000)
dqn_model.save('models/mimic_dqn')

#dqn_model = DQN.load('models/mimic_dqn')

dqn_return, dqn_len = eval_policy(dqn_model, env)