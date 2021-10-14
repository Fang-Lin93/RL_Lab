import gym
import random
from collections import deque
from gym import spaces
from agents import QTableAgent


class WindyGridWorldEnv(gym.Env):
    """
    Exmaple 6.5 in Sutton's textbook
    indices are given as (x, y) not (row, col)
    x: left -> right
    y: up -> down
    """
    metadata = {'render.modes': ['human']}
    N = 0
    S = 1
    E = 2
    W = 3

    def __init__(self, grid_size=(10, 7)):
        super(WindyGridWorldEnv, self).__init__()

        # Size of the 2D-grid
        self.grid_size = grid_size
        self.max_x, self.max_y = self.grid_size[0] - 1, self.grid_size[1] - 1
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.agent_pos = None
        self.goal_pos = [7, 3]
        self.reset()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.grid_size[0],
                                            shape=(2,), dtype=int)

        self.agent = None

    def set_agent(self, agent_):
        self.agent = agent_

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.agent_pos = [0, 3]
        return self.get_obs(), 0, False, 'start'

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

    def step(self, action):
        if action == self.N:
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == self.S:
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.max_y)
        elif action == self.E:
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.max_x)
        elif action == self.W:
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        else:
            raise ValueError(f"Received invalid action={action}")

        # self.prune_pos()
        self.agent_pos[1] = max(self.agent_pos[1] - self.wind[self.agent_pos[0]], 0)

        done_ = (self.agent_pos == self.goal_pos)
        reward_ = 0 if done_ else -1
        info_ = {}

        return self.get_obs(), reward_, done_, info_

    # def prune_pos(self):
    #     self.agent_pos[0] = max(0, min(self.agent_pos[0], self.grid_size[0] - 1))
    #     self.agent_pos[1] = max(0, min(self.agent_pos[1], self.grid_size[1] - 1))

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        grids = [['A' if self.agent_pos == [i, j] else '_' for i in range(self.grid_size[0])]
                 for j in range(self.grid_size[1])]
        grids[self.goal_pos[1]][self.goal_pos[0]] = 'T'  # Terminal States
        grids.append([str(_) for _ in self.wind])
        grids = '\n'.join([' '.join(_) for _ in grids])
        print(grids)

    def get_obs(self):
        return {
            'la': [0, 1, 2, 3],
            'pos': self.agent_pos,
        }


class WindyAgent(QTableAgent):
    """
    discount factor = 1
    use on-line -> train while running
    use off-line -> use replay buffer to train
    """

    def __init__(self, grid_size=(10, 7), n_action_space=4, **kwargs):
        self.grid_size = grid_size

        super().__init__(n_action_space, **kwargs)
        self.name = 'WindyGW'

    def init_table(self):
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                self.Q_table[f'{x}-{y}'] = [0 for _ in range(self.n_action_space)]

    def act(self, obs: dict, **kwargs):
        """
        const reward = -1
        """
        s = self.encode_s(obs)

        if self.training and random.random() < self.eps_greedy:
            a = random.choice(obs['la'])
        else:
            qs = self.Q_table[s] if not self.require_delayed_update() else self.target_Q[s]
            a = qs.index(max(qs))

        if self.training:
            if self.online:
                if self.pre_sa:
                    self.online_train(-1, s, a)
                self.pre_sa = (s, a)
            else:
                self.trajectory += [s, a]

        return a

    @staticmethod
    def encode_s(obs: dict, **kwargs):
        return f'{obs["pos"][0]}-{obs["pos"][1]}'

    def backup(self, payoff=0):
        """
        constant reward = -1 for each step
        """
        if self.online:
            self.online_train(payoff)
        else:
            for i in range(len(self.trajectory)-2, -1, -2):  # (s, a, r, s', a')
                s, a = (self.trajectory[i], self.trajectory[i + 1])

                if self.value_target == 'Q':  # (s, a, s'/G )
                    self.rb.append((s, a, self.trajectory[i + 2] if i + 2 < len(self.trajectory) else payoff))

                elif self.value_target == 'sarsa':  # (s, a, target_value )
                    if i + 3 < len(self.trajectory):
                        tar = self.discount_factor * self.Q_table[self.trajectory[i + 2]][self.trajectory[i + 3]] - 1
                    else:
                        tar = payoff
                    self.rb.append((s, a, tar))

                elif self.value_target == 'mc':
                    self.rb.append((s, a, payoff))
                    payoff = self.discount_factor * payoff - 1

                else:
                    raise ValueError(f'value target should be either td or mc, not {self.value_target}')

    def offline_train(self):
        if self.online:
            return
        if self.rb:
            """
            batch average
            """
            batch_y = {}
            for (s, a, y) in self.rb:
                if (s, a) not in batch_y:
                    batch_y[(s, a)] = [0, 1]  # (value, cnt)
                else:
                    batch_y[(s, a)][1] += 1
                if self.value_target == 'Q' and isinstance(y, str):
                    batch_y[(s, a)][0] += self.discount_factor * max(self.target_Q[y]) - 1
                else:
                    batch_y[(s, a)][0] += y

            for (s, a) in batch_y.keys():
                self.Q_table[s][a] = self.Q_table[s][a] + \
                                     self.lr * (batch_y[(s, a)][0]/batch_y[(s, a)][1] - self.Q_table[s][a])

            if self.value_target == 'Q':
                self.sync_Q()
            else:
                self.rb = deque([], maxlen=self.buffer_size)

    def __str__(self):
        return f'Windy_{self.value_target}_{"online" if self.online else "offline"}'


if __name__ == '__main__':
    """
    off-line/MC methods does not works well if the max_len of episodes is small
    (Reaching Terminal is very sparse and a random walk from S to T is very hard)
    If you wish to try off-line variates or MC, please set max_len larger in Env.run() and use small 'update_freq'
    See Example 6.5 in Sutton's book for details
    """
    from matplotlib import pyplot as plt
    from tqdm import trange

    N_episodes = 200
    max_len = 100000
    update_freq = 1  # for offline only

    env = WindyGridWorldEnv()

    methods = [('Q', True), ('sarsa', True)]
    #  [('Q', True), ('sarsa', True), ('Q', False), ('sarsa', False), ('mc', False)]

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    for (m, online) in methods:
        step, finished_episodes = 0, []
        tag = m + f'_{"online" if online else "offline"}'
        agent = WindyAgent(training=True, value_target=m, online=online, lr=0.5)
        env.set_agent(agent)
        rec_payoff = []
        for episode in trange(N_episodes, desc=tag):
            agent.training = True
            obs, reward, done, info = env.reset()

            payoff = 0
            for _ in range(max_len):
                action = env.agent.act(obs)
                obs, reward, done, info = env.step(action)

                step += 1
                finished_episodes.append(episode)
                if done:
                    break
                payoff += reward

            # R = env.run()

            agent.backup()
            agent.reset()

            if episode % update_freq == 0:
                agent.offline_train()

            rec_payoff.append(payoff)

        agent.save_model()
        """
        training results
        """
        ax[0].plot(range(len(rec_payoff)), rec_payoff, label=tag)
        ax[0].set_ylabel('Avg. score')
        ax[0].set_xlabel('Episodes')
        """
        reproduce results in Example 6.5
        """
        ax[1].plot(range(step), finished_episodes, label=tag)
        ax[1].set_ylabel('Episodes')
        ax[1].set_xlabel('Time steps')

        """
        plot policy
        """
        agent.training = False
        obs, reward, done, info = env.reset()
        path = [obs['pos'][:]]
        act = []
        di = ['^', 'v', '>', '<']
        payoff = 0
        for _ in range(100):
            action = env.agent.act(obs)
            act.append(di[action])
            obs, reward, done, info = env.step(action)
            path.append(obs['pos'][:])
            if done:
                break
            payoff += reward
        grids = [[act[path.index([i, j])] if [i, j] in path[:-1] else '_' for i in range(env.grid_size[0])]
                 for j in range(env.grid_size[1])]
        grids[env.goal_pos[1]][env.goal_pos[0]] = 'T'  # Terminal States
        grids.append([str(_) for _ in env.wind])
        grids = '\n'.join([agent.__str__()] + [' '.join(_) for _ in grids])
        print(grids)

    ax[0].legend()
    ax[1].legend()
    fig.show()

    #
