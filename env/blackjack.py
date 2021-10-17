
import random
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


class BlackJackEnv(object):
    """
    multi-players vs 1 dealer, as Example 5.1
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

