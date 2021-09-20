
import os
import sys
import torch
import json
import threading
from loguru import logger

"""
Buffers, Actors and Processes
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Lock = threading.Lock()


class ReplayBuffer(object):

    def __init__(self, size):
        self.size = size
        self.data = []

    def cla(self):
        self.data = []

    def add(self, data):
        self.data.append(data)

        if self.__len__() > self.size:
            self.data.pop(0)

    def sample_data(self, **kwargs):
        """
        sample ? get all ?
        """
        raise NotImplementedError

    def __get_is_full(self):
        return self.__len__() >= self.size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    is_full = property(__get_is_full)


class Actor(threading.Thread):
    def __init__(self, idx: int, RB: ReplayBuffer, device_=device, **kwargs):
        super().__init__()
        self.idx = idx
        self.device = device_

        self.model = None
        self.init_model()
        self.RB = RB

        self._kwargs = kwargs

    def init_model(self):
        """
        initialize the model (usually with random paras)
        """
        raise NotImplementedError

    def run(self):
        """
        The main function to rollout and get the trajectories
        """

        raise NotImplementedError

    def sync_paras(self, model_paras):
        self.model.load_state_dict(model_paras)

    def __repr__(self):
        return f'Actor-{self.idx}'


class Trainer(object):
    def __init__(self, **kwargs):

        logger.info(kwargs)
        self._kwargs = kwargs

        self.branch = kwargs.get('branch', 0)
        self.save_dir = kwargs.get('save_dir', 'checkpoints')
        self.branch_folder = f'{self.save_dir}/branch_{self.branch}'
        if not os.path.exists(self.branch_folder):
            os.mkdir(self.branch_folder)
        logger.remove()
        logger.add(f'{self.branch_folder}/train.log', rotation="100 MB", level=kwargs.get('logger_level', 'INFO'))
        logger.add(sys.stderr, level=kwargs.get('logger_level', 'INFO'))

        self.target_model = None
        self.RB = None
        self.running_performance = [[], []]

        raise NotImplementedError

    def sync_models(self):
        """
        for multi-actors -> sync models as the target one
        """
        raise NotImplementedError

    def train(self):
        """
        main training loop <- Actor process + update process

        Example:

        s_time = time.time()
        for actor in self.actors:
            actor.start()

        :return:
        """

        raise NotImplementedError

    def update(self):
        """
        updates -> Q, policy gradient, etc.
        """

        raise NotImplementedError

    def evaluate(self):
        """
        to evaluate the current model, and save to self.running_performance
        """

        raise NotImplementedError

    def save_checkpoint(self, i):
        self.target_model.save_model(f'rl_{i}', f'{self.branch_folder}/')

    def load_checkpoint(self, branch=None, load_path=None):
        """
        default to load the checkpoint with the same branch
        """
        if branch is None:
            branch = self.branch

        if load_path is None:
            load_path = f'{self.save_dir}/branch_{branch}/'  # f'{self.branch_folder}/'

        try:
            with open(f'{load_path}performance_{branch}.json', 'r') as file:
                self.running_performance = json.load(file)
            self.target_model.load_model(f'rl_{branch}', load_path)

            logger.info(f'Successfully load the latest checkpoint with {len(self.running_performance[0])} evaluations')
        except Exception as ex:
            self.running_performance = [[], []]
            logger.info(ex)
        self.sync_models()






