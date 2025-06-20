import sys
import os
import gymnasium as gym
from gymnasium import register

# adding path allows us to import from envs.maze
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

register(
    id="maze2dtwogoals-v0",
    entry_point="envs.maze:MazeEnv",
    max_episode_steps=800,
    kwargs={"mode": -1},  # Set to -1 for multimodal (two goals)
)

