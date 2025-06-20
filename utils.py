import numpy as np
import torch.nn as nn
import random
import math
import pickle
import os
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import jax
import flax
import jax.numpy as jnp
import optax
from functools import partial


### For running an experiment ###


class Flags:
    """
    Contains the settings for a batch of experiments.
    """

    required_flags = ['env', 'seed', 'device', 'batch_size', 'n_epochs', 'lr', 'optimizer', 'min_delta', 'patience', 'eval_freq']

    def __init__(self, flags):
        # where flags is a dict w/ all parameters
        for flag in self.required_flags:
            if flag not in flags:
                raise ValueError(f"Must specifcy {flag} in flags")

        self.flags = flags
        for key, val in flags.items():
            setattr(self, key, val)

    def get_flags(self):
        return self.flags
    

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)



### For training VAEs ###


class Annealer:
    """
    Used for annealing a hyperparameter, e.g. KL divergence weight or ε for action selection, during training.
    """
    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = "none"
            self.baseline = 0.0

    def __call__(self, kld):
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == "linear":
            y = self.current_step / self.total_steps
        elif self.shape == "cosine":
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == "logistic":
            exponent = (self.total_steps / 2) - self.current_step
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == "none":
            y = 1.0
        else:
            raise ValueError(
                "Invalid shape for annealing function. Must be linear, cosine, or logistic."
            )
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError(
                "Cyclical_setter method requires boolean argument (True/False)"
            )
        else:
            self.cyclical = value
        return


### For handling datasets ###


class PreferenceDataset(Dataset):
    def __init__(self, pref_dataset):
        self.pref_dataset = pref_dataset

    def __len__(self):
        return len(self.pref_dataset["observations"])

    def __getitem__(self, idx):
        # gets obs1, obs2, and label for a single preference (idx)
        observations = self.pref_dataset["observations"][idx]
        observations_2 = self.pref_dataset["observations_2"][idx]
        labels = self.pref_dataset["labels"][idx]
        return dict(
            observations=observations, observations_2=observations_2, labels=labels
        )

    def get_mode_data(self, batch_size):
        # gets batch_size number of preferences from the dataset
        batch_size = min(batch_size, len(self))
        idxs = np.random.choice(range(len(self)), size=batch_size, replace=False)
        return dict(
            observations=self.pref_dataset["observations"][idxs],
            observations_2=self.pref_dataset["observations_2"][idxs],
        ), batch_size


def get_datasets(query_path, observation_dim, action_dim, batch_size, set_length):
    with open(query_path, "rb") as fp:
        batch = pickle.load(fp)

    batch["observations"] = batch["observations"][..., :observation_dim]
    batch["observations_2"] = batch["observations_2"][..., :observation_dim]
    assert batch["actions"].shape[-1] == action_dim
    if set_length < 0:
        set_length = batch["observations"].shape[1]
    
    eval_data_size = int(0.1 * len(batch["observations"])) # about 10% of data for evaluation
    train_data_size = len(batch["observations"]) - eval_data_size

    train_batch = {
        "observations": batch["observations"][:train_data_size, :set_length],
        "actions": batch["actions"][:train_data_size, :set_length],
        "observations_2": batch["observations_2"][:train_data_size, :set_length],
        "actions_2": batch["actions_2"][:train_data_size, :set_length],
        "labels": batch["labels"][:train_data_size, :set_length],
    }

    eval_batch = {
        "observations": batch["observations"][train_data_size:, :set_length],
        "actions": batch["actions"][train_data_size:, :set_length],
        "observations_2": batch["observations_2"][train_data_size:, :set_length],
        "actions_2": batch["actions_2"][train_data_size:, :set_length],
        "labels": batch["labels"][train_data_size:, :set_length],
    }

    train_dataset = PreferenceDataset(train_batch)
    eval_dataset = PreferenceDataset(eval_batch)
    kwargs = {"num_workers": 1, "pin_memory": True} # not parallelized, fasater CPU-to-GPU memory transfer
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = DataLoader(
        dataset=eval_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )

    _, _, len_query, obs_dim = batch["observations"].shape
    return (
        train_loader,
        test_loader,
        train_dataset,
        eval_dataset,
        set_length,
        len_query,
        obs_dim,
    )




### For logging metrics during training/eval ###


class Logger:

    def __init__(self, save_dir, env, log_freq, n_epochs):
        self.save_dir = save_dir
        self.env = env
        self.log_freq = log_freq
        self.n_epochs = n_epochs

        self.losses = []
        self.accuracies = []
        self.rewards = []

    def log_metrics(self, metrics):
        self.train_losses.append(metrics['loss'])
        self.train_accuracies.append(metrics['accuracy'])
        self.train_rewards.append(metrics['reward'])

    def graph_metrics(self, save=True, filename='metrics'):
        fig, axs = plt.subplots(ncols=4, figsize=(12, 5))
        axs[0].set_title("Loss")
        axs[0].plot(self.losses)

        axs[1].set_title("Accuracy")
        axs[1].plot(self.accuracies)

        axs[2].set_title("Reward")
        axs[2].plot(self.rewards)

        axs[3].set_title("Episode Length")
        axs[3].plot(self.env.length_queue)

        if save:
            plt.savefig(os.path.join(self.save_dir, f"{filename}.png"))

            

