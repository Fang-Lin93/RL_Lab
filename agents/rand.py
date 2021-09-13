
import random


class RandomAgent(object):

    def __init__(self):
        self.name = 'random_agent'

    def step(self, state: dict):
        return random.choice(state['la'])

