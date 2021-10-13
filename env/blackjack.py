import gym
import random
from loguru import logger
from collections import deque
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
            acts = [a.act(obs, i) for i, a in enumerate(self.agents)]
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
            self.payoff = [-1]*self.num_players
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
            'scores': [p.score for p in self.players],
            'usable_ace': [int(p.usable_ace > 0) for p in self.players],
            'finished': [p.finished for p in self.players]
        }
        return obs


class QTableAgent(object):
    """
    discount factor = 1
    use on-line -> train while running
    use off-line -> use replay buffer to train
    """
    def __init__(self, **kwargs):
        self.name = 'Q_table agent'
        self.eps_greedy = kwargs.get('eps_greedy', 0.1)
        self.training = kwargs.get('training', False)
        self.value_target = kwargs.get('value_target', 'mc')  # either 'mc', 'sarsa'(td), 'Q'
        self.buffer_size = kwargs.get('buffer_size', 10000)
        self.on_line = kwargs.get('on_line', False)  # on-line/off-line training
        self.lr = kwargs.get('lr', 0.1)

        if self.on_line:
            if self.value_target == 'mc':
                logger.warning('MC-target is not acceptable for on-line learning. The learning mode is set to off-line')
                self.on_line = False

        # on-line training
        self.pre_sa = None

        # off-line training
        self.rb = deque([], maxlen=self.buffer_size)  # replay buffer
        self.trajectory = []

        # Q tables
        self.Q_table = {}  # policy Q
        self.target_Q = {}  # only for Q-learning (delay updates)
        self.init_table()
        if self.value_target == 'Q':
            self.sync_Q()

    def reset(self):
        self.pre_sa = ()
        self.trajectory = []

    def sync_Q(self):
        self.target_Q = {k: v[:] for k, v in self.Q_table.items()}

    def init_table(self):
        for _d_c in range(1, 11):
            for _s in range(12, 22):
                self.Q_table[f'{_d_c}-{_s}-0'] = [random.uniform(-1, 1), random.uniform(-1, 1)]
                self.Q_table[f'{_d_c}-{_s}-1'] = [random.uniform(-1, 1), random.uniform(-1, 1)]

    def act(self, obs: dict, player_id=0):
        if not obs['finished'][player_id]:
            s = self.encode_s(obs, player_id)

            if self.training and random.random() < self.eps_greedy:
                a = random.choice(obs['la'])
            else:
                qs = self.Q_table[s]
                a = int(qs[1] > qs[0])

            if self.training:
                if self.on_line:
                    if self.pre_sa:
                        self.on_line_train(0, s, a)
                    self.pre_sa = (s, a)
                else:
                    self.trajectory += [s, a]

            return a
        return

    @staticmethod
    def encode_s(obs: dict, player_id):
        return f'{obs["dear_hand"]}-{obs["scores"][player_id]}-{obs["usable_ace"][player_id]}'

    def off_line_train(self):
        if self.on_line:
            return
        if self.rb:
            for (s, a, y) in self.rb:
                if self.value_target == 'Q' and isinstance(y, str):
                    self.Q_table[s][a] = self.Q_table[s][a] + self.lr * (max(self.target_Q[y]) - self.Q_table[s][a])
                else:
                    self.Q_table[s][a] = self.Q_table[s][a] + self.lr * (y - self.Q_table[s][a])

            if self.value_target == 'Q':
                self.sync_Q()
            else:
                self.rb = deque([], maxlen=self.buffer_size)

    def on_line_train(self, r=0, s=None, a=None):
        if not self.pre_sa:
            return
        p_s, p_a = self.pre_sa
        if s is None:
            y = r
        else:
            y = r + max(self.Q_table[s]) if self.value_target == 'Q' else r + self.Q_table[s][a]

        self.Q_table[p_s][p_a] = self.Q_table[p_s][p_a] + self.lr * (y - self.Q_table[p_s][p_a])

    def backup(self, payoff):
        """
        backup the final reward and train if necessary
        """
        if self.on_line:
            self.on_line_train(payoff)
        else:
            for i in range(0, len(self.trajectory), 2):
                s, a = (self.trajectory[i], self.trajectory[i + 1])

                if self.value_target == 'Q':  # (s, a, s'/G )
                    self.rb.append((s, a, self.trajectory[i + 2] if i + 2 < len(self.trajectory) else payoff))

                elif self.value_target == 'sarsa':  # (s, a, target_value )
                    if i + 3 < len(self.trajectory):
                        tar = self.Q_table[self.trajectory[i + 2]][self.trajectory[i + 3]]
                    else:
                        tar = payoff
                    self.rb.append((s, a, tar))

                elif self.value_target == 'mc':
                    self.rb.append((s, a, payoff))

                else:
                    raise ValueError(f'value target should be either td or mc, not {self.value_target}')


if __name__ == '__main__':
    """
    Test some tabular methods
    Reproducing the results in Sutton's book (Example 5.1)
    """
    from agents.rand import RandomAgent
    from tqdm import trange
    from matplotlib import pyplot as plt

    def test(agent_, n=10000):
        res = []
        agent_.training = False
        env_ = BlackJackEnv(num_players=1, show_log=False)
        env_.set_agents([agent_])
        for _ in range(n):
            env_.reset()
            res.append(env_.run()[0])
        return sum(res)/len(res)

    N_episodes = 10000
    update_freq = 1000
    test_freq = N_episodes // 50

    env = BlackJackEnv(num_decks=-1, num_players=1, show_log=False)

    methods = [('Q', True), ('sarsa', True), ('Q', False), ('sarsa', False), ('mc', False)]

    plt.figure(figsize=(10, 8))

    for (m, online) in methods:
        tag = m+f'_{"online"if online else "offline"}'
        agent = QTableAgent(training=True, value_target=m, on_line=online)
        env.set_agents([agent])
        rec_payoff = []
        for episode in trange(N_episodes, desc=tag):

            if episode % test_freq == 0:
                rec_payoff.append(test(agent))

            agent.training = True
            env.reset()
            R = env.run()[0]
            agent.backup(R)
            agent.reset()

            if R == 0:
                assert env.payoff[0] == 0

            if episode % update_freq == 0:
                agent.off_line_train()

        plt.plot(rec_payoff, label=tag)

    plt.legend()
    plt.show()
