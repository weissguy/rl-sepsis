import gymnasium as gym
import torch
from collections import defaultdict
import numpy as np
import envs # import is necessary to register the custom envs

from utils import Flags, Annealer, Logger, set_random_seed, get_datasets
from models import MLPModel, CategoricalModel, MeanVarianceModel, MLPClassifier
from vae import VAEModel, VAEClassifier


save_dir = 'results'


# TODO: create a default flags, and store configs for models in a json file

flags = Flags({
    # Environment
    'env': 'maze2dtwogoals-v0',
    'seed': 42,
    'device': 'cuda',
    'set_size': 1,
    'biased_mode': 'grid',  # for maze envs
    
    # Training
    'batch_size': 256,
    'n_epochs': 500,
    'lr': 1e-3,
    'optimizer': 'Adam',
    'min_delta': 3e-4,
    'patience': 10,
    'eval_freq': 100,
    
    # Model
    'model_type': 'MLP',
    'hidden_dim': 256,
    
    # Categorical Model
    'num_atoms': 10,
    'r_min': 0,
    'r_max': 1,
    'entropy_coeff': 0.1,
    
    # Mean Variance Model
    'variance_penalty': 0.0,
    
    # VAE Model
    'latent_dim': 32,
    'kl_weight': 1.0,
    'learned_prior': False,
    'flow_prior': False,  # TODO what is flow prior?
    'use_annealing': False,
    'annealer_baseline': 0.0,
    'annealer_type': 'cosine',
    'annealer_cycles': 4,
    'reward_scaling': 1.0,
    
    # Dataset
    'dataset_path': 'datasets/icu_data.csv',
    
})


def main():

    # create gym environment
    env = gym.make(flags.env)
    env.action_space.seed(flags.seed)
    env.observation_space.seed(flags.seed)
    set_random_seed(flags.seed) # TODO is this reproducible? why after setting?
    
    # Unwrap the environment to access MazeEnv methods directly
    env = env.unwrapped

    if hasattr(env, 'seed'):
        env.seed(flags.seed)
    if hasattr(env, "reward_observation_space"):
        observation_dim = env.reward_observation_space.shape[0] # TODO what has a reward observation space?
    else:
        observation_dim = env.observation_space['observation'].shape[0]
    if 'maze' in flags.env:
        env.set_biased_mode(flags.biased_mode) # Now we can call this directly
    action_dim = env.action_space.shape[0]

    # load train, test, and eval datasets
    (
        train_loader,
        test_loader,
        train_dataset,
        eval_dataset,
        len_set,
        len_query,
        obs_dim,
    ) = get_datasets(
        flags.dataset_path,
        observation_dim,
        action_dim,
        flags.batch_size,
        flags.set_size,
    )

    # create logger
    logger = Logger(save_dir, env, flags.eval_freq, flags.n_epochs)

    # create model
    if flags.model_type == "MLP":
        reward_model = MLPModel(obs_dim, flags.hidden_dim)
    elif flags.model_type == "Categorical":
        reward_model = CategoricalModel(
            input_dim=obs_dim,
            hidden_dim=flags.hidden_dim,
            n_atoms=flags.num_atoms,
            r_min=flags.r_min,
            r_max=flags.r_max,
            entropy_coeff=flags.entropy_coeff,
        )
    elif flags.model_type == "MeanVar":
        reward_model = MeanVarianceModel(
            input_dim=obs_dim,
            hidden_dim=flags.hidden_dim,
            variance_penalty=flags.variance_penalty,
        )
    elif "VAE" in flags.model_type:
        annealer = None
        if flags.use_annealing:
            annealer = Annealer(
                total_steps=flags.n_epochs // flags.annealer_cycles,
                shape=flags.annealer_type,
                baseline=flags.annealer_baseline,
                cyclical=flags.annealer_cycles > 1,
            )
        if flags.model_type == "VAEClassifier":
            reward_model = VAEClassifier
            decoder_input = 2 * obs_dim + flags.latent_dim
        else:
            reward_model = VAEModel
            decoder_input = obs_dim + flags.latent_dim
        reward_model = reward_model(
            encoder_input=len_set * (2 * observation_dim * len_query + 1),
            decoder_input=decoder_input,
            latent_dim=flags.latent_dim,
            hidden_dim=flags.hidden_dim,
            annotation_size=len_set,
            size_segment=len_query,
            kl_weight=flags.kl_weight,
            learned_prior=flags.learned_prior,
            flow_prior=flags.flow_prior, 
            annealer=annealer,
            reward_scaling=flags.reward_scaling,
        )
    elif flags.model_type == "MLPClassifier":
        reward_model = MLPClassifier(obs_dim, flags.hidden_dim)
    else:
        raise NotImplementedError('model type not defined')
    
    # create optimizer
    if flags.optimizer == 'Adam':
        optimizer = torch.optim.Adam(reward_model.parameters(), lr=flags.lr)
    elif flags.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(reward_model.parameters(), lr=flags.lr)
    else:
        raise NotImplementedError('optimizer not defined')
    
    device = flags.device
    reward_model = reward_model.to(device)
    best_criteria = None

    # actual training loop
    for epoch in range(flags.n_epochs):
        metrics = defaultdict(list)
        metrics['epoch'] = epoch

        for batch in train_loader:
            optimizer.zero_grad()
            observations = batch['observations'].to(device).float()
            observations_2 = batch['observations_2'].to(device).float()
            labels = batch['labels'].to(device).float()
            loss, batch_metrics = reward_model(observations, observations_2, labels)
            loss.backward()
            optimizer.step()

            # update metrics for logging

        # evaluate (only on certain epochs)
        if epoch % flags.eval_freq == 0:
            for batch in test_loader:
                with torch.no_grad():
                    observations = batch['observations'].to(device).float()
                    observations_2 = batch['observations_2'].to(device).float()
                    labels = batch['labels'].to(device).float()
                    loss, batch_metrics = reward_model(observations, observations_2, labels)
                    metrics |= batch_metrics

                    # update metrics for logging
                    logger.log_metrics(metrics)

            # NOTE: omitting debugging plots for now

            criteria = np.mean(metrics["loss"])

            if best_criteria is None or criteria < best_criteria:
                best_criteria = criteria
                torch.save(reward_model, save_dir + f"/best_model.pt")

            # NOTE: omitting early stopping for now

            if (flags.model_type == "VAEClassifier" or flags.model_type == "VAEModel") and flags.use_annealing:
                annealer.step()

            # torch.save(reward_model, save_dir + f"/model_{epoch}.pt")

    # log at end of training
    logger.graph_metrics(save=True, filename=f"train_metrics_{flags.model_type}")
    


if __name__ == "__main__":
    main()