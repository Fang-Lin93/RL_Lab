
from gym import spaces


class GridWorldEnv(object):
    """
    indices are given as (x, y)
    x: left -> right
    y: up -> down
    """
    N = 0
    S = 1
    E = 2
    W = 3

    def __init__(self, grid_size):
        super(GridWorldEnv, self).__init__()

        # Size of the 2D-grid
        self.grid_size = grid_size
        self.max_x, self.max_y = self.grid_size[0] - 1, self.grid_size[1] - 1
        self.agent = None
        self.agent_pos = None
        self.goal_pos = None
        self.reset()
        self.action_space = spaces.Discrete(4)

    def set_agent(self, agent_):
        self.agent = agent_

    def run(self, max_len=100):
        obs, reward, done, info = self.reset()
        payoff = 0
        for _ in range(max_len):
            action = self.agent.act(obs)
            obs, reward, done, info = self.step(action)
            if done:
                break
            payoff += reward
        return payoff

    def reset(self):
        """
        Example:

        self.agent_pos = [2, 2]
        return self.agent_pos
        """
        raise NotImplementedError()

    def step(self, action: int):
        self.move(action)

        raise NotImplementedError()

    def move(self, a: int):
        if a == self.N:
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif a == self.S:
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.max_y)
        elif a == self.E:
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.max_x)
        elif a == self.W:
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        else:
            raise ValueError(f"Received invalid action={a}")

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        grids_ = self.get_raw_grid()
        grids_[self.agent_pos[1]][self.agent_pos[0]] = 'A'

        grids_ = '\n'.join([' '.join(_) for _ in grids_])
        print(grids_)

    def get_raw_grid(self):
        grids_ = [['_'] * self.grid_size[0] for _ in range(self.grid_size[1])]
        grids_[self.goal_pos[1]][self.goal_pos[0]] = 'T'
        return grids_


class BoardGameEnv(object):
    pass

