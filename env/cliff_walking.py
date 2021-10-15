
from gym import spaces
from env.basic import GridWorldEnv


class CliffWalkingEnv(GridWorldEnv):
    """
    Example 6.6 in Sutton's textbook
    cliff: y = 3 & 0 < x < 11
    """

    def __init__(self, grid_size=(12, 4)):
        super(CliffWalkingEnv, self).__init__(grid_size)

        self.goal_pos = [11, 3]
        self.reset()
        self.action_space = spaces.Discrete(4)

    def set_agent(self, agent_):
        self.agent = agent_

    def reset(self):

        self.agent_pos = [0, 3]
        return self.get_obs(), 0, False, 'start'

    def step(self, action):
        self.move(action)

        done_, reward_, info_ = False, -1, {}
        if self.agent_pos[1] == 3 and 0 < self.agent_pos[0] < 11:
            done_, reward_, info_ = True, -100, 'dead'

        elif self.agent_pos == self.goal_pos:
            done_, reward_, info_ = True, 0, 'Goal reached'

        return self.get_obs(), reward_, done_, info_

    def get_raw_grid(self):

        grids_ = [['_'] * self.grid_size[0] for _ in range(self.grid_size[1])]
        grids_[self.goal_pos[1]][self.goal_pos[0]] = 'T'
        grids_[3][1:11] = ['░'] * 10
        return grids_

    def get_obs(self):
        return {
            'la': [0, 1, 2, 3],
            'pos': self.agent_pos,
        }

    def __str__(self):
        return f'Cliff_Walking_{self.grid_size}'


if __name__ == '__main__':
    env = CliffWalkingEnv()

    obs = env.reset()
    env.render()

    n_steps = 10
    for step in range(n_steps):
        act = env.action_space.sample()
        print(f"Step {step + 1} -> act = {act}")
        obs, reward, done, info = env.step(act)
        print('obs=', obs, 'reward=', reward, 'done=', done)
        env.render()
        if done:
            print(info)
            break
