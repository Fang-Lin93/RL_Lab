
from gym import spaces
from env.basic import GridWorldEnv


class WindyGridWorldEnv(GridWorldEnv):
    """
    Example 6.5 in Sutton's textbook
    """

    def __init__(self, grid_size=(10, 7)):
        super(WindyGridWorldEnv, self).__init__(grid_size)

        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.goal_pos = [7, 3]
        self.reset()
        self.action_space = spaces.Discrete(4)

    def set_agent(self, agent_):
        self.agent = agent_

    def reset(self):

        self.agent_pos = [0, 3]
        return self.get_obs(), 0, False, 'start'

    def step(self, action):
        self.move(action)

        self.agent_pos[1] = max(self.agent_pos[1] - self.wind[self.agent_pos[0]], 0)
        done_ = (self.agent_pos == self.goal_pos)
        reward_ = 0 if done_ else -1
        info_ = {}

        return self.get_obs(), reward_, done_, info_

    # def render(self, mode='console'):
    #     if mode != 'console':
    #         raise NotImplementedError()
    #     grids_ = [['A' if self.agent_pos == [i, j] else '_' for i in range(self.grid_size[0])]
    #               for j in range(self.grid_size[1])]
    #     grids_[self.goal_pos[1]][self.goal_pos[0]] = 'T'  # Terminal States
    #     grids_.append([str(_) for _ in self.wind])
    #     grids_ = '\n'.join([' '.join(_) for _ in grids_])
    #     print(grids_)
    #     return grids_

    def get_raw_grid(self):
        grids_ = [['_'] * self.grid_size[0] for _ in range(self.grid_size[1])]
        grids_[self.goal_pos[1]][self.goal_pos[0]] = 'T'  # Terminal States
        grids_.append([str(_) for _ in self.wind])
        return grids_

    def get_obs(self):
        return {
            'la': [0, 1, 2, 3],
            'pos': self.agent_pos,
        }

    def __str__(self):
        return f'Windy_GW_{self.grid_size}'
