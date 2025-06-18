import minari
import gymnasium as gym
import numpy as np
import math



### Generic helpers ###

def factor_int(n):
    val = math.ceil(math.sqrt(n))
    val2 = int(n / val)
    while val2 * val != float(n):
        val -= 1
        val2 = int(n / val)
    return val, val2, n


def gridify_state(state):
    return (int(round(state[0])), int(round(state[1])))


# TODO combine MazeEnv and CustomMazeEnv (from maze_utils.py)

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

EXPLORATION_ACTIONS = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT: (1, 0)}

# Custom maze specification constants
WALL = '#'
EMPTY = 'O'
REWARD = 'R'
START = 'S'

class MazeEnv(gym.Env):

    def __init__(self, mode=-1, hidden_eval=False, bonus=False, maze_spec=None):
        super().__init__()

        dataset = minari.load_dataset('D4RL/pointmaze/medium-v2') # add download=True if not installed
        self.env = dataset.recover_environment()

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,)
        )
        self._max_episode_steps = self.env._max_episode_steps

        self.mode = mode
        #     self.train_goals = np.array([(7, 1), (7, 10), (1,1), (1, 10), (3,8), (5,4), (1, 4), (5, 10), (4, 1), (5, 7)])
        #     self.goals = self.train_goals
        #     self.test_goals = np.array([(3, 10), (7, 4), (1, 6), (3, 1), (7, 8)])
        # else:
        self.goals = np.array([(7, 1), (7, 10)])
        self.relabel_offline_reward = True
        self.is_multimodal = mode < 0
        self.biased_mode = None
        if not self.is_multimodal:
            self.env.set_target(self.goals[mode])

        if maze_spec is None:
            # Default medium maze specification
            self.maze_spec = [
                "############",
                "#OOOO#OOOOO#",
                "#O##O#O#O#O#",
                "#OOOOOO#OOO#",
                "#O####O###O#",
                "#OO#O#OOOOO#",
                "##O#O#O#O###",
                "#OO#OOO#OOO#",
                "############"
            ]
        else:
            # Parse string specification
            self.maze_spec = maze_spec.split('\\') if isinstance(maze_spec, str) else maze_spec
        
        self.height = len(self.maze_spec)
        self.width = len(self.maze_spec[0])
        self.grid = self._parse_maze()

        self.num_states = self.width * self.height
        self.num_actions = 4  # UP, DOWN, LEFT, RIGHT

        self.qmatrixes = [self.get_qmatrix(goal, self.get_obs_grid()) for goal in self.goals]


    @property
    def target(self):
        return self.env.unwrapped._target

    def get_dataset(self):
        return self.env.get_dataset()

    def reset(self):
        return self.env.reset()
    
    # def set_training_goals(self):
    #     self.goals = self.train_goals
    #     self.qmatrixes = self.train_qmatrixes
    
    # def set_eval_goals(self):
    #     self.goals = self.test_goals
    #     self.qmatrixes = self.test_qmatrixes

    def step(self, action):
        # Compute shaped reward
        obs, reward, done, info = self.env.step(action)
        # Override environment termination
        success = np.linalg.norm(obs[:2] - self.target) < 0.5
        if success:
            done = True # terminate the episode if the agent reaches the goal
        info["actual_reward"] = reward
        reward = success
        return obs, reward, done, info

    def compute_reward(self, obs, mode):
        # Setting mode to random if not provided
        if self.mode < 0:
            if mode < 0:
                mode = np.random.randint(len(self.goals))
            mode = mode
        else:
            mode = self.mode
        qmatrix, r_min, r_max = self.qmatrixes[mode]
        obs_xy = obs[:, :, :2]
        reward = self.get_reward(qmatrix, obs_xy)
        reward = (reward - r_min) / (r_max - r_min)
        return reward
    
        
    def _parse_maze(self):
        """Parse maze specification into a 2D grid."""
        grid = []
        for row in self.maze_spec:
            grid_row = []
            for cell in row:
                if cell == WALL:
                    grid_row.append(WALL)
                elif cell == REWARD:
                    grid_row.append(REWARD)
                elif cell == START:
                    grid_row.append(START)
                else:
                    grid_row.append(EMPTY)
            grid.append(grid_row)
        return grid
    
    def is_wall(self, x, y):
        """Check if position (x, y) is a wall."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return self.grid[y][x] == WALL
    
    def is_reward(self, x, y):
        """Check if position (x, y) is a reward location."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.grid[y][x] == REWARD
    
    def get_grid_state(self, x, y):
        """Get grid state at position (x, y)."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return WALL
        return self.grid[y][x]
    
    def get_obs_grid(self):
        obs = np.mgrid[0:8:50j, 0:11:50j]
        obs = obs.reshape(obs.shape[0], -1).T
        return obs

    def xy_to_idx(self, xy):
        """Convert (x, y) coordinates to linear index."""
        x, y = int(round(xy[0])), int(round(xy[1]))
        # Clamp coordinates to valid bounds
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        return y * self.width + x
    
    def idx_to_xy(self, idx):
        """Convert linear index to (x, y) coordinates."""
        y = idx // self.width
        x = idx % self.width
        return (x, y)
    
    
    def transition_matrix(self):
        """
        Create transition matrix for the maze.
        Returns a matrix of shape (num_states * num_actions, num_states)
        where T[s*a, s'] = P(s'|s, a)
        """

        T = np.zeros((self.num_states * self.num_actions, self.num_states))
        
        for state_idx in range(self.num_states):
            x, y = self.idx_to_xy(state_idx)
            
            for action in range(self.num_actions):
                # Get next position based on action
                dx, dy = EXPLORATION_ACTIONS[action]
                next_x, next_y = x + dx, y + dy
                
                # If next position is a wall, stay in current position
                if self.is_wall(next_x, next_y):
                    next_state_idx = state_idx
                else:
                    next_state_idx = self.xy_to_idx((next_x, next_y))
                
                # Set transition probability
                T[state_idx * self.num_actions + action, next_state_idx] = 1.0
        
        return T
    
    def reward_matrix(self):
        """
        Create reward matrix for the maze.
        Returns a matrix of shape (num_states, num_actions)
        """

        R = np.zeros((self.num_states, self.num_actions))
        
        for state_idx in range(self.num_states):
            x, y = self.idx_to_xy(state_idx)
            if self.is_reward(x, y):
                R[state_idx, :] = 1.0
        
        return R

    def plot_gt(self, wandb_log=False):
        import matplotlib.pyplot as plt
        import wandb

        xv, yv = np.meshgrid(
            np.linspace(*(0, 8), 100), np.linspace(*(0, 11), 100), indexing="ij"
        )
        points = np.concatenate([xv.reshape(-1, 1), yv.reshape(-1, 1)], axis=1)[None]
        r = [self.compute_reward(points, mode) for mode in range(self.get_num_modes())]
        fig, axs = plt.subplots(1, self.get_num_modes(), figsize=(self.get_num_modes()*10, 8))
        if self.get_num_modes() == 1:
            axs_flat = [axs]
        else:
            axs_flat = axs.flatten()
        for i, ax in enumerate(axs_flat):
            r_hat = self.apply_walls(r[i][0], points[0])
            sc = ax.scatter(points[0, :, 0], points[0, :, 1], c=r_hat)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("r(s)")
            self.plot_goals(ax)
        plt.tight_layout()
        if wandb_log:
            return wandb.Image(fig)
        else:
            print("Saving reward plot")
            plt.savefig("reward_plot")
        plt.close(fig)

    def plot_goals(self, ax):
        self.plot_walls(ax, self.get_obs_grid())
        for g in self.goals:
            ax.scatter(g[0], g[1], s=100, c="black", marker="o")

    def set_biased_mode(self, mode):
        self.biased_mode = mode

    def get_biased_data(self, set_len):
        if self.biased_mode == "grid":
            w, l, _ = factor_int(set_len * 2)
            obs = np.mgrid[0 : 8 : w * 1j, 0 : 11 : l * 1j]
            obs = obs.reshape(obs.shape[0], -1).T
        elif self.biased_mode == "random":
            obs = np.random.uniform(0, 1, (set_len * 2, 2)) * np.array([8, 11])
        elif self.biased_mode == "equal":
            obs_y = np.random.uniform(5, 7, size=(2 * set_len,))
            obs_x = np.random.uniform(1, 7, size=(2 * set_len,))
            obs = np.stack([obs_x, obs_y], axis=1)
        else:
            raise ValueError("Invalid biased mode")
        # idxs = np.random.permutation(np.arange(obs.shape[0]))
        return obs[:set_len], np.copy((obs[set_len:])[::-1])

    def get_goals(self):
        return (self.goals / np.array([8, 11])) * 50

    def render(self, mode="rgb_array"):
        return self.env.render(mode)

    ## Functions to handle multimodality
    def get_num_modes(self):
        if self.is_multimodal:
            return len(self.goals)
        return 1

    def sample_mode(self):
        if self.is_multimodal:
            return np.random.randint(len(self.goals))
        return self.mode

    def reset_mode(self):
        self.set_mode(self.sample_mode())

    def set_mode(self, mode):
        if self.is_multimodal:
            self.mode = mode
            self.env.set_target(self.goals[mode])


    def q_iteration(self, num_itrs=50, discount=0.99, warmstart_q=None, policy=None):
        """
        Perform tabular Q-iteration to solve for an optimal policy.
        """

        if warmstart_q is None:
            q_fn = np.zeros((self.num_states, self.num_actions))
        else:
            q_fn = warmstart_q

        for _ in range(num_itrs):
            if policy is None:
                v_fn = np.max(q_fn, axis=1)
            else:
                v_fn = np.sum(q_fn * policy, axis=1)
            # dynamic programming update on bellman equation
            num_states, num_actions = self.num_states, self.num_actions
            new_q = self.reward_matrix() + discount * self.transition_matrix().dot(v_fn).reshape(num_states, num_actions)
            q_fn = new_q

        return q_fn


    def get_qmatrix(self, goal, obs):
        # Set goal as reward location
        goal_grid = gridify_state(goal)
        # Clamp goal coordinates to valid bounds
        goal_x = max(0, min(goal_grid[0], self.width - 1))
        goal_y = max(0, min(goal_grid[1], self.height - 1))
        self.grid[goal_y][goal_x] = REWARD
        
        q_values = self.q_iteration(num_itrs=500, discount=0.99)
        
        obs_grid = np.array([gridify_state(o) for o in obs])
        obs_id = [self.xy_to_idx(o) for o in obs_grid]
        r = np.max(q_values[obs_id], axis=1)
        rs = np.array([r[i] for i in range(len(r)) if not self.is_wall(obs_grid[i][0], obs_grid[i][1])])
        return (q_values, rs.min(), rs.max())
    

    def get_reward(self, q_values, obs): # B x S x T x 2
        obs_original_shape = obs.shape
        obs = obs.reshape(-1, 2)
        obs_grid = np.array([gridify_state(o) for o in obs])
        obs_id = [self.xy_to_idx(o) for o in obs_grid]
        r = np.max(q_values[obs_id], axis=1).reshape(obs_original_shape[:-1])
        return r


    def apply_walls(self, reward, obs):
        for i in range(obs.shape[0]):
            obs_grid = gridify_state(obs[i])
            if self.is_wall(obs_grid[0], obs_grid[1]):
                reward[i] = 0
        return reward

    def plot_walls(self, ax, obs):
        walls = []
        for i in range(obs.shape[0]):
            obs_grid = gridify_state(obs[i])
            if self.is_wall(obs_grid[0], obs_grid[1]):
                walls.append(obs[i])
        walls = np.array(walls)
        ax.scatter(walls[:, 0], walls[:, 1], c="black")
        return ax


    HARDEST_MAZE_TEST ='############\\'+\
                    '#OOOO#OOOOO#\\'+\
                    '#O##O#O#O#O#\\'+\
                    '#OOOOOO#OOO#\\'+\
                    '#O####O###O#\\'+\
                    '#OO#O#OOOOO#\\'+\
                    '##O#O#O#O###\\'+\
                    '#OO#OOO#OOO#\\'+\
                    '############'

    def get_qmatrix_antmaze(self, goal, obs):
        return self.get_qmatrix(goal, obs, maze_spec=self.HARDEST_MAZE_TEST)
