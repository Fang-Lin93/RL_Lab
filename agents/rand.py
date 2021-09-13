
import random


class RandomAgent(object):

    def __init__(self):
        self.name = 'random_agent'

    def step(self, obs_seq, la):
        return random.choice(la)

