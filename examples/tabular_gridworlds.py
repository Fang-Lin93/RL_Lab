
import random
import torch
from torch import nn
from env.basic import GridWorldEnv
from agents import TabularGWAgent
from matplotlib import pyplot as plt
from tqdm import trange

"""
 off-line/MC methods does not works well if the max_len of episodes is small for windyGW
 (Reaching Terminal is very sparse and a random walk from S to T is very hard)
 If you wish to try off-line variates or MC on windyGW , please set max_len larger in Env.run() and use small 'update_freq'
 See Example 6.5 in Sutton's book for details
 """


class LinearGWAgent(TabularGWAgent):
    """
    online only
    """

    def __init__(self, input_c, n_action_space=4, **kwargs):
        super().__init__(n_action_space=n_action_space, **kwargs)

        self.name = 'linear_GW'
        self.input_c = input_c
        self.eps_greedy = kwargs.get('eps_greedy', 0.1)
        self.policy_model = self.init_policy()

    def init_table(self):
        return {}

    def init_policy(self):
        return nn.Linear(self.input_c, self.n_action_space)

    def policy2table(self):
        with torch.no_grad():
            m = self.policy_model
        # TODO

    def act(self, obs: dict, **kwargs):
        """
        const reward = -1
        """
        s = self.encode_s(obs)

        if self.training and random.random() < self.eps_greedy:
            a = random.choice(obs['la'])
        else:
            qs = self.predict(s)
            a = qs.argmax(dim=-1)

        if self.training:
            if self.online:
                if self.pre_sa:
                    self.online_train(-1, s, a)
                self.pre_sa = (s, a)

        return a

    def predict(self, x):
        with torch.no_grad():
            return self.policy_model(x)

    def encode_s(self, obs: dict, **kwargs):
        """
        use one-hot
        """
        vec = [0] * self.input_c
        vec[self.grid_size[0]*obs["pos"][1] + obs["pos"][0]] = 1  # TODO
        return torch.FloatTensor(vec)

    def backup(self, payoff=0):
        """
        backup final immediate reward (payoff)
        constant reward = -1 for each step
        """
        if self.online and self.training:
            self.online_train(payoff)

    def online_train(self, r=0, s=None, a=None):
        if not self.pre_sa:
            return
        self.policy_model.train()
        p_s, p_a = self.pre_sa
        if s is None:
            y = r
        else:
            y = r + self.discount_factor * self.policy_model(s).max().item() if self.value_target == 'Q' \
                else r + self.discount_factor * self.policy_model(s)[a].item()

        self.policy_model.zero_grad()
        loss = (self.policy_model(p_s)[p_a] - y) ** 2
        loss.backward()
        # gradient descent
        with torch.no_grad():
            for p in self.policy_model.parameters():
                p -= 2 * self.lr * p.grad
        self.policy_model.eval()
        self.policy2table()

    def __str__(self):
        return f'{self.value_target}_{"online" if self.online else "offline"}'


def run_grid(World: GridWorldEnv, tabular=True):
    N_episodes = 1000
    max_len = 100
    update_freq = 1  # for offline only

    env = World

    methods = [('Q', True), ('sarsa', True)]
    # methods = [('Q', True), ('sarsa', True), ('Q', False), ('sarsa', False), ('mc', False)]
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    for (m, online) in methods:
        step, finished_episodes = 0, []
        tag = m + f'_{"online" if online else "offline"}'
        if tabular:
            agent = TabularGWAgent(grid_size=env.grid_size,
                                   training=True, value_target=m, online=online, lr=0.5)
        else:
            input_c = env.grid_size[0]*env.grid_size[1]
            agent = LinearGWAgent(input_c=input_c, grid_size=env.grid_size, training=True,
                                  value_target=m, online=online, lr=0.001)
        print(agent)
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
                payoff += reward
                if done:
                    break

            agent.backup(reward)
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
            payoff += reward
            if done:
                break

        grids = env.get_raw_grid()
        for (i, j), a in zip(path, act):
            grids[j][i] = a

        grids = '\n'.join([env.__str__() + '_' +
                           agent.__str__() + f'_reward={payoff}'] + [' '.join(_) for _ in grids])
        print(grids)

    ax[0].set_title(env.__str__())
    ax[0].legend()
    ax[1].legend()
    fig.show()


if __name__ == '__main__':
    """
    why the algorithm not always finds the path during the testing phase?
    (randomness ?)
    """
    random.seed(0)
    from env import WindyGridWorldEnv, CliffWalkingEnv
    run_grid(WindyGridWorldEnv(), tabular=False)
    run_grid(CliffWalkingEnv(), tabular=False)
