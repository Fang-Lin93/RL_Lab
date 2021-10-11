import gym
import random
from gym import spaces


def hand_sum(hand: list):
    """
    -1 means bust
    """
    cnt_ace = hand.count(1)
    raw_score = sum([c if c != 1 else 0 for c in hand])
    raw_score += cnt_ace
    if raw_score > 21:
        return -1
    # find the maximal hand sum <= 21
    if cnt_ace > 0:
        for change in range(cnt_ace):
            if raw_score + 10 <= 21:
                raw_score += 10
            else:
                break

    return raw_score


class Player(object):
    def __init__(self, player_id):
        self.player_id = player_id
        self.hand = []
        self.finished = False
        self.is_bust = False

    def check(self):
        score = self.score()
        if score < 0:
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

        self.hand = [self.next_card(), self.next_card()]  # deal 2 cards to the dealer
        random.shuffle(self.deck)

    def next_card(self):
        if not self.deck:
            return
        if self.infinite_deck:
            return random.choice(self.deck)
        return self.deck.pop(0)

    def act(self):
        while (0 < hand_sum(self.hand) < 17) and self.deck:
            self.hand.append(self.next_card())

    def is_bust(self):
        return self.score() > 21

    def score(self):
        return hand_sum(self.hand)


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
        self.dealer = None
        self.action_space = spaces.Discrete(2)
        self.players = []

    def reset(self):
        # hide one card
        self.dealer = Dealer(self.num_decks)
        self.players = [Player(_) for _ in range(self.num_players)]
        for p in self.players:
            p.hand.append(self.dealer.next_card())
            p.hand.append(self.dealer.next_card())
            p.check()  # natural?
        return (self.dealer.hand[1:], [p.hand[1:] for p in self.players]), \
            [0] * self.num_players, self.all_finished(), 'Reset'

    def step(self, actions):
        """
        actions of players
        :return: obs, reward, done?, info
        """
        done_ = False
        for a, p in zip(actions, self.players):
            if not p.finished:
                if a == self.HIT:
                    p.hand.append(self.dealer.next_card())
                else:
                    p.finished = True
            p.check()

        reward_ = [-1 if p.is_bust else 0 for p in self.players]

        if reward_.count(-1) == self.num_players:
            return (self.dealer.hand[1:], [p.hand[1:] for p in self.players]), reward_, True, 'All bust'

        if self.all_finished():
            self.dealer.act()
            # calculate final score
            if self.dealer.is_bust():
                reward_ = [-1 if p.is_bust else 1 for p in self.players]
            else:
                reward_ = [-1 if p.is_bust or (p.score() < self.dealer.score())
                           else int(p.score() > self.dealer.score()) for p in self.players]
            done_ = True

        return (self.dealer.hand[1:], [p.hand[1:] for p in self.players]), reward_, done_, 'Finished'

    def all_finished(self):
        return all(p.finished for p in self.players)

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print('===================================')
        for i, p in enumerate(self.players):
            print(f'Player_{i}: {p.hand} score={p.score()} (bust={p.is_bust}) finished={p.finished}')
        print(f'Dealer: {self.dealer.hand} score={self.dealer.score()} (bust={self.dealer.is_bust()})')
        print('===================================')

    def close(self):
        pass


if __name__ == '__main__':
    env = BlackJackEnv(num_players=2)

    obs = env.reset()
    env.render()

    # print(env.observation_space)
    # print(env.action_space)
    # print(env.action_space.sample())

    n_steps = 20
    for step in range(n_steps):
        acts = [env.action_space.sample() for _ in range(env.num_players)]
        print(f"Step {step + 1}:")
        print(f" act = {acts}")
        obs, reward, done, info = env.step(acts)
        print('obs=', obs, 'reward=', reward, 'done=', done)
        env.render()
        if done:
            print("Game finished!", "reward=", reward)
            break
