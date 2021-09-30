
import random


class RandomAgent(object):

    def __init__(self):
        self.name = 'random_agent'

    @staticmethod
    def act(state: dict):
        return random.choice(state['la'])

