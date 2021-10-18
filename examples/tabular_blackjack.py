import random
import seaborn as sns
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt
from env import BlackJackEnv
from agents import QTableAgent


class BlackJackAgent(QTableAgent):
    """
    discount factor = 1
    use on-line -> train while running
    use off-line -> use replay buffer to train
    """

    def __init__(self, n_action_space=2, **kwargs):
        self.rand_init = kwargs.get('rand_init', False)

        super().__init__(n_action_space, **kwargs)
        self.name = 'BJ'

    def init_table(self):
        """
        (dealer card) - (my score) - (usable ace)
        initially, sticks only when score >= 20 as in the textbook
        or using random
        """

        for _d_c in range(1, 11):
            for _s in range(12, 22):
                if self.rand_init:
                    self.Q_table[f'{_d_c}-{_s}-0'] = [random.uniform(-1, 1), random.uniform(-1, 1)]
                    self.Q_table[f'{_d_c}-{_s}-1'] = [random.uniform(-1, 1), random.uniform(-1, 1)]
                else:
                    self.Q_table[f'{_d_c}-{_s}-0'] = [-0.1 if _s >= 20 else 0.1, 0]
                    self.Q_table[f'{_d_c}-{_s}-1'] = [-0.1 if _s >= 20 else 0.1, 0]

    def act(self, obs: dict, **kwargs):
        player_id = kwargs.get('player_id')
        if not obs['finished'][player_id]:
            s = self.encode_s(obs, player_id=player_id)

            if self.training and random.random() < self.eps_greedy:
                a = random.choice(obs['la'])
            else:
                qs = self.Q_table[s] if not self.require_delayed_update() else self.target_Q[s]
                a = int(qs[1] > qs[0])

            if self.training:
                if self.online:
                    if self.pre_sa:
                        self.online_train(0, s, a)
                    self.pre_sa = (s, a)
                else:
                    self.trajectory += [s, a]

            return a
        return

    @staticmethod
    def encode_s(obs: dict, **kwargs):
        player_id = kwargs.get('player_id')
        return f'{obs["dear_hand"]}-{obs["scores"][player_id]}-{obs["usable_ace"][player_id]}'

    def backup(self, payoff):
        """
        backup the final reward and train if necessary
        """
        if self.online:
            self.online_train(payoff)
        else:
            for i in range(len(self.trajectory) - 2, -1, -2):  # (s, a, s', a')
                s, a = (self.trajectory[i], self.trajectory[i + 1])

                if self.value_target == 'Q':  # (s, a, 0, s' )
                    if i + 2 < len(self.trajectory):
                        self.rb.append((s, a, 0, self.trajectory[i + 2]))
                    else:
                        self.rb.append((s, a, payoff, None))

                elif self.value_target == 'sarsa':  # (s, a, target_value )
                    if i + 3 < len(self.trajectory):
                        tar = self.discount_factor * self.Q_table[self.trajectory[i + 2]][self.trajectory[i + 3]]
                    else:
                        tar = payoff
                    self.rb.append((s, a, tar, None))

                elif self.value_target == 'mc':
                    self.rb.append((s, a, payoff, None))
                    payoff *= self.discount_factor

                else:
                    raise ValueError(f'value target should be either td or mc, not {self.value_target}')

    def plot_policy(self):
        self.sync_Q()

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

        self.sync_Q()
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


class FixAgent(BlackJackAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'BJ_fix_strategy'

    def init_table(self):
        for _d_c in range(1, 11):
            for _s in range(12, 22):
                self.Q_table[f'{_d_c}-{_s}-0'] = [-1, -1]  # pessimistic
                self.Q_table[f'{_d_c}-{_s}-1'] = [-1, -1]

    def act(self, obs: dict, **kwargs):
        player_id = kwargs.get('player_id')
        if not obs['finished'][player_id]:
            s = self.encode_s(obs, player_id=player_id)
            if obs['scores'][player_id] >= 20:
                a = 0
            else:
                a = 1
            self.trajectory += [s, a]
            return a


def train_policies(N_episodes=100000, N_decks=1, N_players=1):
    update_freq = 1000  # for offline only
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

    methods = [('Q', True), ('sarsa', True), ('Q', False), ('sarsa', False), ('mc', False)]

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    for (m, online) in methods:
        tag = m + f'_{"online" if online else "offline"}'
        agent = BlackJackAgent(training=True, value_target=m, online=online)
        env.set_agents([agent])
        rec_payoff, rec_win = [], []
        for episode in trange(N_episodes, desc=tag):
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
            if episode % update_freq == 0:
                agent.offline_train()

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


"""
Reproducing the results in Sutton's book (Example 5.1)
"""


def value_surface():
    fix_agent = FixAgent(training=True, value_target='mc', lr=0.1, buffer_size=10000)

    env = BlackJackEnv(num_decks=-1, num_players=1, show_log=False)
    env.set_agents([fix_agent])
    for episode in trange(500000, desc='Fix policy'):
        env.reset()
        R = env.run()[0]
        fix_agent.backup(R)
        fix_agent.reset()

        if episode % 1000 == 0:
            fix_agent.offline_train()
    fix_agent.plot_value()


if __name__ == '__main__':
    train_policies()
    value_surface()
