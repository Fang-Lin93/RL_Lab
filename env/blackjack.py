import gym
import random
from collections import deque
from tqdm import trange
from matplotlib import pyplot as plt
from agents import QTableAgent
from gym import spaces


def hand_sum(hand: list):
    """
    find the maximal (if ace) hand sum and usable ace
    """
    cnt_ace, usable_ace = hand.count(1), 0
    raw_score = sum([c for c in hand if c != 1])
    raw_score += cnt_ace
    if raw_score > 21:
        return raw_score, usable_ace
    # find the maximal hand sum <= 21
    if cnt_ace > 0:
        for _ in range(cnt_ace):
            if raw_score + 10 <= 21:
                usable_ace += 1
                raw_score += 10
            else:
                break

    return raw_score, usable_ace


class Player(object):
    def __init__(self, player_id):
        self.player_id = player_id
        self.hand = []

        self.finished = False
        self.is_bust = False
        self.score = 0
        self.usable_ace = 0

    def check(self):
        self.score, self.usable_ace = hand_sum(self.hand)
        if self.score > 21:
            self.is_bust = True
            self.finished = True
        #
        # if self.score == 21:
        #     self.finished = True


class Dealer(object):
    """
    -1 gives infinite number of decks
    id=1,  2,  3,  4, ...,  9, 10
    cd=A,  2,  3,  4, ...,  9, (10/J/Q/K)
    cnt=x4, x4, x4, ..., x4, x16 -> 9x4 + 16
    """

    def __init__(self, num_decks: int = 1):
        self.num_decks = num_decks if num_decks > 0 else 1
        self.infinite_deck = False
        if self.num_decks < 0:
            self.infinite_deck = True

        self.deck = [j for i in [[c] * 4 * self.num_decks
                                 if c < 10 else [c] * 16 * self.num_decks for c in range(1, 11)] for j in i]

        random.shuffle(self.deck)

        self.hand = self.next_cards(2)  # deal 2 cards to the dealer
        self.score = 0
        self.is_bust = False
        self.check()

    def next_cards(self, num=1):
        if not (self.infinite_deck or self.deck):
            return
        if self.infinite_deck:
            return [random.choice(self.deck) for _ in range(num)]
        return [self.deck.pop(0) for _ in range(num)]

    def act(self):
        while hand_sum(self.hand)[0] < 17 and self.deck:
            self.hand += self.next_cards()

        self.check()

    def check(self):
        self.score, _ = hand_sum(self.hand)
        if self.score > 21:
            self.is_bust = True


class BlackJackEnv(gym.Env):
    """
    multi-players vs 1 dealer
    """
    metadata = {'render.modes': ['console']}

    STICK = 0
    HIT = 1

    def __init__(self, **kwargs):
        super(BlackJackEnv, self).__init__()

        self.num_decks = kwargs.get('num_decks', 1)
        self.num_players = kwargs.get('num_players', 1)  # not counting the dealer here
        self.show_log = kwargs.get('show_log', False)
        self.dealer = None
        self.payoff = [0] * self.num_players
        self.action_space = spaces.Discrete(2)
        self.players = []
        self.agents = []

    def set_agents(self, agents):
        self.agents = agents

    def reset(self):
        self.payoff = [0] * self.num_players
        self.dealer = Dealer(self.num_decks)
        self.players = [Player(_) for _ in range(self.num_players)]
        for p in self.players:
            p.hand += self.dealer.next_cards(2)
            p.check()  # natural?
            while p.score < 12:  # score < 12, it will always hit
                p.hand += self.dealer.next_cards()
                p.check()

        done = self.check_all_finished()
        return self.get_obs(), self.payoff, done, 'Reset'

    def run(self, max_steps=100):

        obs, reward, done, info = self.reset()

        for step in range(max_steps):
            if self.show_log:
                self.render()
            acts = [a.act(obs, player_id=i) for i, a in enumerate(self.agents)]
            if self.show_log:
                print(f"Step {step + 1}: actions={acts}")
            obs, reward, done, info = self.step(acts)
            if done:
                break

        if self.show_log:
            self.render()

        return self.payoff

    def step(self, actions: list):
        """
        actions of players
        :return: obs, reward, done, info
        """

        for a, p in zip(actions, self.players):
            if not p.finished:
                if a == self.HIT:
                    p.hand += self.dealer.next_cards()
                elif a == self.STICK:
                    p.finished = True
                else:
                    raise ValueError(f'Illegal action {a} from player-{p.player_id}')
            p.check()

        if sum([p.is_bust for p in self.players]) == self.num_players:
            self.payoff = [-1] * self.num_players
            return self.get_obs(), self.payoff, True, 'All bust'

        done = self.check_all_finished()
        return self.get_obs(), self.payoff, done, 'Finished'

    def check_all_finished(self):
        """
        perform dealer's fixed strategy
        """
        if all(p.finished for p in self.players):
            self.dealer.act()
            # calculate final score
            if self.dealer.is_bust:
                self.payoff = [-1 if p.is_bust else 1 for p in self.players]  # player bust earlier than the dealer
            else:
                self.payoff = [-1 if (p.is_bust or (p.score < self.dealer.score))
                               else int(p.score > self.dealer.score) for p in self.players]
            return True
        return False

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print('===================================')
        for i, p in enumerate(self.players):
            print(f'Player_{i}: {p.hand} score={p.score} (bust={p.is_bust}) finished={p.finished}')
        print(f'Dealer: {self.dealer.hand} score={self.dealer.score} (bust={self.dealer.is_bust})')
        print(f'Payoff = {self.payoff}')
        print('===================================')

    def get_obs(self):
        obs = {
            'la': [0, 1],
            'dear_hand': self.dealer.hand[1],
            'scores':  [p.score for p in self.players],  # [sum(p.hand) for p in self.players]
            'usable_ace': [int(p.usable_ace > 0) for p in self.players],
            'finished': [p.finished for p in self.players]
        }
        return obs


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
                    batch_y[(s, a)][0] += self.discount_factor * max(self.target_Q[y])
                else:
                    batch_y[(s, a)][0] += y

            for (s, a) in batch_y.keys():
                self.Q_table[s][a] = self.Q_table[s][a] + self.lr * (batch_y[(s, a)][0]/batch_y[(s, a)][1] - self.Q_table[s][a])

            if self.value_target == 'Q':
                self.sync_Q()
            else:
                self.rb = deque([], maxlen=self.buffer_size)

    def backup(self, payoff):
        """
        backup the final reward and train if necessary
        """
        if self.online:
            self.online_train(payoff)
        else:
            for i in range(len(self.trajectory)-2, -1, -2):  # (s, a, s', a')
                s, a = (self.trajectory[i], self.trajectory[i + 1])

                if self.value_target == 'Q':  # (s, a, s'/G )
                    self.rb.append((s, a, self.trajectory[i + 2] if i + 2 < len(self.trajectory) else payoff))

                elif self.value_target == 'sarsa':  # (s, a, target_value )
                    if i + 3 < len(self.trajectory):
                        tar = self.discount_factor * self.Q_table[self.trajectory[i + 2]][self.trajectory[i + 3]]
                    else:
                        tar = payoff
                    self.rb.append((s, a, tar))

                elif self.value_target == 'mc':
                    self.rb.append((s, a, payoff))
                    payoff *= self.discount_factor

                else:
                    raise ValueError(f'value target should be either td or mc, not {self.value_target}')

    def plot_policy(self):
        import seaborn as sns
        self.sync_Q()

        opt_no_ace = [[0]*10 for _ in range(10)]  # [dealer card][my score]
        opt_ace = [[0]*10 for _ in range(10)]

        for k, v in self.Q_table.items():
            dealer_c, score, ace = [int(_) for _ in k.split('-')]
            if ace:
                opt_ace[score-12][dealer_c-1] = int(v[1] > v[0])
            else:
                opt_no_ace[score-12][dealer_c-1] = int(v[1] > v[0])

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
                v_ace[dealer_c-1][score-12] = max(v)
            else:
                v_no_ace[dealer_c-1][score-12] = max(v)

        return self.plot_surface(v_ace, self.__str__() + 'With Ace'), \
            self.plot_surface(v_no_ace, self.__str__() + 'Without Ace')

    def plot_surface(self, surf, desc=''):
        import numpy as np
        X, Y = np.meshgrid(range(10), range(10))
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_title(self.__str__() +desc)
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


if __name__ == '__main__':
    """
    Find optimal tabular methods: Q, sarsa, mc
    """
    N_episodes = 100000
    update_freq = 1000  # for offline only
    test_freq = N_episodes // 50  # evaluate the agent after some episodes
    N_decks = 1
    N_players = 1

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

        agent.save_model()
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

    #
    """
    Reproducing the results in Sutton's book (Example 5.1)
    """

    class FixAgent(BlackJackAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.name = 'fix_strategy'

        def init_table(self):
            for _d_c in range(1, 11):
                for _s in range(12, 22):
                    self.Q_table[f'{_d_c}-{_s}-0'] = [-1, -1]
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

    fix_agent = FixAgent(training=True, value_target='mc',  lr=0.1, buffer_size=10000)

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





