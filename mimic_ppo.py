from envs.mimic import MIMICEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


#GAMMA = 0.99
#LR = 0.01

env = make_vec_env('mimic-v0', n_envs=4)

model = PPO('MlpPolicy', 'mimic-v0', verbose=1).learn(total_timesteps=1000)
model.save('mimic_ppo2')

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render('human')