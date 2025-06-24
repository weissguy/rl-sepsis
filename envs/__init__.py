import sys
import os
from gymnasium import register

# adding path allows us to import from envs.maze
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


register(
    id="mimic-v0",
    entry_point="envs.mimic:MIMICEnv",
    max_episode_steps=1000,
    kwargs={},
)


register(
    id="maze2dtwogoals-v0",
    entry_point="envs.maze:MazeEnv",
    max_episode_steps=800,
    kwargs={"mode": -1},  # Set to -1 for multimodal (two goals)
)