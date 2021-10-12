import gym
import random
from collections import deque
from gym import spaces


def hand_sum(hand: list):
    """
    -1 means bust
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
        self.usable_ace = 0

    def check(self):
        score, self.usable_ace = self.score()
        if score > 21:
            self.is_bust = True
            self.finished = True

        if score == 21:
            self.finished = True

    def score(self):
        return hand_sum(self.hand)


class Dealer(object):
    """
     id=0,  1,  2,  3, ...,  8, 9
     fa=A,  2,  3,  4, ...,  9, (10/J/Q/K)
    cnt=x4, x4, x4, ..., x4, x16 -> 9x4 + 16
    """

    def __init__(self, num_decks: int = 1):
        self.num_decks = num_decks
        self.infinite_deck = False
        if self.num_decks < 0:
            self.infinite_deck = True

        self.deck = [j for i in [[c] * 4 * num_decks if c < 10 else [c] * 16 * num_decks for c in range(1, 11)] for
                     j in i]

        random.shuffle(self.deck)

        self.hand = self.next_cards(2)  # deal 2 cards to the dealer

    def next_cards(self, num=1):
        if not self.deck:
            return
        if self.infinite_deck:
            return [random.choice(self.deck) for _ in range(num)]
        return [self.deck.pop(0) for _ in range(num)]

    def act(self):
        while self.score() < 17 and self.deck:
            self.hand += self.next_cards()

    def is_bust(self):
        return self.score() > 21

    def score(self):
        return hand_sum(self.hand)[0]


class BlackJackEnv(gym.Env):
    """
    multi-players vs dealer
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
        # hide one card
        self.payoff = [0] * self.num_players
        self.dealer = Dealer(self.num_decks)
        self.players = [Player(_) for _ in range(self.num_players)]
        for p in self.players:
            p.hand += self.dealer.next_cards(2)
            p.check()  # natural?
        return self.get_obs(), self.payoff, self.check_all_finished(), 'Reset'

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

    def step(self, actions):
        """
        actions of players
        :return: obs, reward, done?, info
        """

        for a, p in zip(actions, self.players):
            if not p.finished:
                if a == self.HIT:
                    p.hand += self.dealer.next_cards()
                else:
                    p.finished = True
            p.check()

        if sum([p.is_bust for p in self.players]) == self.num_players:
            self.payoff = [-1]*self.num_players
            return self.get_obs(), self.payoff, True, 'All bust'

        done = self.check_all_finished()
        return self.get_obs(), self.payoff, done, 'Finished'

    def check_all_finished(self):
        if all(p.finished for p in self.players):
            self.dealer.act()
            # calculate final score
            if self.dealer.is_bust():
                self.payoff = [-1 if p.is_bust else 1 for p in self.players]
            else:
                self.payoff = [-1 if (p.is_bust or (p.score()[0] < self.dealer.score()))
                               else int(p.score()[0] > self.dealer.score()) for p in self.players]
            return True
        return False

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print('===================================')
        for i, p in enumerate(self.players):
            print(f'Player_{i}: {p.hand} score={p.score()} (bust={p.is_bust}) finished={p.finished}')
        print(f'Dealer: {self.dealer.hand} score={self.dealer.score()} (bust={self.dealer.is_bust()})')
        print(f'Payoff = {self.payoff}')
        print('===================================')

    def close(self):
        pass

    def get_obs(self):
        obs = {
            'la': [0, 1],
            'dear_hand': self.dealer.hand[1],
            'scores': [hand_sum(p.hand)[0] for p in self.players],
            'usable_ace': [int(p.usable_ace > 0) for p in self.players],
            'finished': [p.finished for p in self.players]
        }
        return obs


class QTableAgent(object):
    def __init__(self, **kwargs):
        self.name = 'Q_table agent for blackjack'
        self.eps_greedy = kwargs.get('eps_greedy', 0.1)
        self.training = kwargs.get('training', False)
        self.value_target = kwargs.get('value_target', 'mc')  # either 'mc', 'sarsa (td)' 'Q'
        self.buffer_size = kwargs.get('buffer_size', 1000)  # either 'mc', 'sarsa (td)' 'Q'
        self.lr = kwargs.get('lr', 0.1)

        self.Q_table = {}  # target-Q
        self.policy = {}  # running
        if self.value_target == 'Q':
            self.sync_Q()

        # for training
        self.rb = deque([], maxlen=self.buffer_size)  # replay buffer
        self.trajectory = []

        self.init_table()

    def sync_Q(self):
        self.policy = {k: v[:] for k, v in self.Q_table.items()}

    def init_table(self):
        for _d_c in range(1, 11):
            for _s in range(4, 21):
                self.Q_table[f'{_d_c}-{_s}-0'] = self.Q_table[f'{_d_c}-{_s}-1'] = [0, 0]
                if self.value_target == 'Q':
                    self.policy[f'{_d_c}-{_s}-0'] = self.Q_table[f'{_d_c}-{_s}-1'] = [0, 0]

    def act(self, obs: dict, player_id=0):
        if not obs['finished'][player_id]:
            s = self.encode_s(obs, player_id)

            if random.random() < self.eps_greedy:
                action = random.choice(obs['la'])
            else:
                qs = self.Q_table[s]
                action = int(qs[1] > qs[0])

            if self.training:
                self.trajectory += [s, action]
            return action

    @staticmethod
    def encode_s(obs: dict, player_id):
        return f'{obs["dear_hand"]}-{obs["scores"][player_id]}-{obs["usable_ace"][player_id]}'

    def train(self):
        for (s, a, y) in self.rb:
            if self.value_target == 'Q' and isinstance(y, str):
                self.Q_table[s][a] = self.Q_table[s][a] - self.lr * (self.Q_table[s][a] - max(self.policy[y]))
            else:
                self.Q_table[s][a] = self.Q_table[s][a] - self.lr * (self.Q_table[s][a] - y)

        if self.value_target == 'Q':
            self.sync_Q()
        else:
            self.rb = deque([], maxlen=self.buffer_size)

    def backup(self, payoff):
        """
        discount factor = 1
        """
        for i in range(0, len(self.trajectory), 2):
            s, a = (self.trajectory[i], self.trajectory[i+1])

            if self.value_target == 'Q':  # (s, a, s'/G )
                self.rb.append((s, a, self.trajectory[i+2] if i+2 < len(self.trajectory) else payoff))

            elif self.value_target == 'sarsa':  # (s, a, s'/G )
                if i + 3 < len(self.trajectory):
                    tar = self.Q_table[self.trajectory[i+2]][self.trajectory[i+3]]
                else:
                    tar = payoff

                self.rb.append((s, a, tar))

            elif self.value_target == 'mc':
                self.rb.append((s, a, payoff))

            else:
                raise ValueError(f'value target should be either td or mc, not {self.value_target}')

        self.trajectory = []


if __name__ == '__main__':
    """
    Reproducing the results in Sutton's book (Example 5.1)
    """
    from agents import RandomAgent

    env = BlackJackEnv(num_players=1, show_log=False)

    agent = QTableAgent(training=True, eps_greedy=1, value_target='sarsa')
    env.set_agents([agent])

    from tqdm import trange

    rec_payoff = []

    for _ in trange(10000):
        env.reset()
        R = env.run()[0]
        rec_payoff.append(R)
        agent.backup(R)

        if R == 0:
            assert env.payoff[0] == 0
            # env.render()

        if len(agent.rb) > 100:
            agent.train()
