import gym
import random
import pyglet
from gym import spaces


class GridWorldEnv(gym.Env):
    """
    as the examples in David Silver lectures
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['human']}
    # Define constants for clearer code
    N = 0
    S = 1
    E = 2
    W = 3

    def __init__(self, grid_size=(5, 5)):
        super(GridWorldEnv, self).__init__()

        # Size of the 2D-grid
        self.grid_size = grid_size
        self.agent_pos = None
        self.reset()

        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=0, high=self.grid_size[0],
                                            shape=(2,), dtype=int)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        while True:
            self.agent_pos = [random.randint(0, self.grid_size[0]-1), random.randint(0, self.grid_size[1]-1)]
            if self.agent_pos != [0, 0] and self.agent_pos != [i-1 for i in self.grid_size]:
                break

        return [self.agent_pos]

    def step(self, action):
        if action == self.N:
            self.agent_pos[1] += 1
        elif action == self.S:
            self.agent_pos[1] -= 1
        elif action == self.E:
            self.agent_pos[0] += 1
        elif action == self.W:
            self.agent_pos[0] -= 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        self.agent_pos[0] = max(0, min(self.agent_pos[0], self.grid_size[0] - 1))
        self.agent_pos[1] = max(0, min(self.agent_pos[1], self.grid_size[1] - 1))

        # Account for the boundaries of the grid
        # self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        # Are we at the left of the grid?
        done = bool(self.agent_pos == [0, 0] or self.agent_pos == [i-1 for i in self.grid_size])

        reward = 0 if done else -1

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        # return np.array([self.agent_pos]).astype(np.float32), reward, done, info
        return [self.agent_pos], reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        grids = [['A' if self.agent_pos == [i, j] else '_' for i in range(self.grid_size[0])]
                 for j in range(self.grid_size[1])]
        # grids = '\n'.join([' '.join(['A' if self.agent_pos == [i, j] else '_' for i in range(self.grid_size[0])])
        #                    for j in range(self.grid_size[1])])
        grids[0][0] = grids[-1][-1] = 'T'  # Terminal States
        grids = '\n'.join([' '.join(_) for _ in grids])
        print(grids)

        return grids


if __name__ == '__main__':
    env = GridWorldEnv(grid_size=(5, 5))

    obs = env.reset()
    env.render()

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    # Hardcoded best agent: always go left!
    n_steps = 20
    for step in range(n_steps):
        print("Step {}".format(step + 1))
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print('obs=', obs, 'reward=', reward, 'done=', done)
        env.render()
        if done:
            print("Goal reached!")
            break

