import random
import torch
from torch import nn
import seaborn as sns
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt
from env import BlackJackEnv
from agents import QTableAgent


class LinearBJAgent(QTableAgent):
    """
    I borrow the tabular agent and using model functions
    rewrite online/offline train
    Only give examples of Q learning and sarsa for online training.

    The off-line follows tabular methods easily.
    """

    def __init__(self, input_c=22, n_action_space=2, **kwargs):
        super().__init__(n_action_space=n_action_space, **kwargs)

        self.name = 'linear_BJ'
        self.input_c = input_c
        self.eps_greedy = kwargs.get('eps_greedy', 0.1)
        self.policy_model = self.init_policy()

    def init_policy(self):
        return nn.Linear(self.input_c, self.n_action_space)

    def init_table(self):
        return {}

    def policy2table(self):
        with torch.no_grad():
            for _d_c in range(1, 11):
                for _s in range(12, 22):
                    vec = [0] * 20
                    vec[_d_c - 1] = 1
                    vec[_s - 2] = 1
                    self.Q_table[f'{_d_c}-{_s}-0'] = self.policy_model(torch.FloatTensor(vec + [1, 0])).tolist()
                    self.Q_table[f'{_d_c}-{_s}-1'] = self.policy_model(torch.FloatTensor(vec + [0, 1])).tolist()

    def act(self, obs: dict, **kwargs):
        player_id = kwargs.get('player_id')
        if not obs['finished'][player_id]:
            s = self.encode_s(obs, player_id=player_id)

            if self.training and random.random() < self.eps_greedy:
                a = random.choice(obs['la'])
            else:
                qs = self.predict(s)
                a = qs.argmax(dim=-1)

            if self.training:
                if self.online:
                    if self.pre_sa:
                        self.online_train(0, s, a)
                    self.pre_sa = (s, a)
            return a
        return

    def predict(self, x):
        with torch.no_grad():
            return self.policy_model(x)

    def online_train(self, r=0, s=None, a=None):
        if not self.pre_sa:
            return
        self.policy_model.train()

        with torch.no_grad():
            p_s, p_a = self.pre_sa
            if s is None:
                y = r
            else:
                y = r + self.discount_factor * self.policy_model(s).max() if self.value_target == 'Q' \
                    else r + self.discount_factor * self.policy_model(s)[a]

        self.policy_model.zero_grad()
        loss = (self.policy_model(p_s)[p_a] - y) ** 2
        loss.backward()

        # gradient descent
        with torch.no_grad():
            for p in self.policy_model.parameters():
                p -= 2 * self.lr * p.grad
        self.policy_model.eval()

    @staticmethod
    def encode_s(obs: dict, **kwargs):
        """
        use one-hot
        """
        player_id = kwargs.get('player_id')
        vec = [0] * 22
        vec[obs["dear_hand"] - 1] = 1
        vec[obs["scores"][player_id] - 2] = 1
        vec[20 + obs["usable_ace"][player_id]] = 1
        return torch.FloatTensor(vec)

    def backup(self, payoff):
        """
        backup the final reward and train if necessary
        """
        if self.online and self.training:
            self.online_train(payoff)

    def plot_policy(self):
        self.policy2table()
        opt_no_ace = [[0] * 10 for _ in range(10)]  # [dealer card][my score]
        opt_ace = [[0] * 10 for _ in range(10)]
        for k, v in self.Q_table.items():
            dealer_c, score, ace = [int(_) for _ in k.split('-')]
            if ace:
                opt_ace[score - 12][dealer_c - 1] = int(v[1] > v[0])
            else:
                opt_no_ace[score - 12][dealer_c - 1] = int(v[1] > v[0])

        fig, ax = plt.subplots(2, 1, figsize=(8, 14))
        for p, n in zip([opt_ace, opt_no_ace], ['With Ace', 'Without Ace']):
            sub_ax = ax[int(n == 'Without Ace')]
            ax_ = sns.heatmap(p, ax=sub_ax)
            ax_.set_title(self.__str__() + n)
            ax_.invert_yaxis()
            sub_ax.set_yticklabels(range(12, 22))
            sub_ax.set_ylabel('Hand Score')
            sub_ax.set_xticklabels(['A'] + [str(_) for _ in range(2, 11)])
            sub_ax.set_xlabel('Dealer Show')

        fig.show()

    def plot_value(self):
        self.policy2table()
        v_no_ace = [[0] * 10 for _ in range(10)]
        v_ace = [[0] * 10 for _ in range(10)]
        for k, v in self.target_Q.items():
            dealer_c, score, ace = [int(_) for _ in k.split('-')]
            if ace:
                v_ace[dealer_c - 1][score - 12] = max(v)
            else:
                v_no_ace[dealer_c - 1][score - 12] = max(v)
        return self.plot_surface(v_ace, self.__str__() + 'With Ace'), \
            self.plot_surface(v_no_ace, self.__str__() + 'Without Ace')

    @staticmethod
    def plot_surface(surf, desc=''):
        X, Y = np.meshgrid(range(10), range(10))
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_title(desc)
        ax.plot_surface(X, Y, np.array(surf), linewidth=10, antialiased=False, alpha=0.5)
        ax.set_zlim(-1, 1)
        ax.view_init(30, 140)
        ax.set_xticks(range(10))
        ax.set_xticklabels(range(12, 22))
        ax.set_xlabel('Hand Score')
        ax.set_yticks(range(10))
        ax.set_yticklabels(['A'] + [str(_) for _ in range(2, 11)])
        ax.set_ylabel('Dealer Show')
        fig.show()
        return fig

    def __str__(self):
        return f'{self.value_target}_{"online" if self.online else "offline"}'


def train_policies(N_episodes=50000, N_decks=1, N_players=1):

    test_freq = N_episodes // 50  # evaluate the agent after some episodes

    def test(agent_, n=1000):
        res = []
        agent_.training = False
        env_ = BlackJackEnv(num_decks=N_decks, num_players=N_players, show_log=False)
        env_.set_agents([agent_])
        for _ in range(n):
            env_.reset()
            res.append(env_.run()[0])
        return sum(res) / len(res), res.count(1) / len(res)

    env = BlackJackEnv(num_decks=N_decks, num_players=N_players, show_log=False)

    methods = ['Q', 'sarsa']

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    for m in methods:
        tag = m
        agent = LinearBJAgent(training=True, value_target=m, online=True, lr=0.001)
        env.set_agents([agent])
        rec_payoff, rec_win = [], []
        for episode in trange(N_episodes, desc=tag + '_linear'):
            if episode % test_freq == 0:
                sco_, win_ = test(agent)
                rec_payoff.append(sco_)
                rec_win.append(win_)

            agent.training = True
            env.reset()
            R = env.run()[0]
            agent.backup(R)
            agent.reset()

            if R == 0:
                assert env.payoff[0] == 0

        # agent.save_model()
        agent.plot_policy()

        plt_x = [test_freq * _ for _ in range(len(rec_payoff))]
        ax[0].plot(plt_x, rec_payoff, label=tag)
        ax[1].plot(plt_x, rec_win, label=tag)
        ax[0].set_ylabel('Avg. Score')
        ax[1].set_ylabel('Win Rate')
        ax[1].set_xlabel('Episodes')

    ax[0].legend()
    ax[1].legend()
    fig.show()


if __name__ == '__main__':
    train_policies()
