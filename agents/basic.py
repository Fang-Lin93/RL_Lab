import os
import sys
import torch
import json
import threading
from .networks.naive import FC_BN, CNN
from loguru import logger

"""
Buffers, Actors and Processes
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Lock = threading.Lock()


class ReplayBuffer(object):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = []

    def cla(self):
        self.data = []

    def add(self, data):
        """
             sample ? get all ?
             """
        self.data.append(data)
        if len(self.data) > self.capacity:
            self.data.pop(0)

    def sample(self, **kwargs):
        """
        sample ? get all ?
        """
        raise NotImplementedError

    def __get_is_full(self):
        return self.__len__() >= self.capacity

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __str__(self):
        return f'ReplayBuffer-{len(self)}/{self.capacity}'

    is_full = property(__get_is_full)


class BaseAgent(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', '')

        # replay buffer
        self.trajectory = []
        self.training = kwargs.get('training', True)
        self.rb = None

        # general training paras
        self.history_len = kwargs.get('history_len', 5)
        self.batch_size = kwargs.get('batch_size', 128)
        self.gamma = kwargs.get('gamma', 0.99)
        self.max_grad_norm = kwargs.get('max_grad_norm', 10)
        self.lr = kwargs.get('lr', 0.0001)
        self.eps = kwargs.get('eps', 1e-5)
        self.disable_byte_norm = kwargs.get('disable_byte_norm', False)
        self.game = kwargs.get('game', 'game')

        # model paras
        self.input_c = kwargs.get('input_c')
        self.n_act = kwargs.get('n_act')
        self.input_rgb = kwargs.get('input_rgb')
        self.lstm = kwargs.get('lstm', False)
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.n_layers = kwargs.get('n_layers', 5)

        if self.lstm:
            self.model_in_channel = 3 if self.input_rgb else self.input_c
        else:
            self.model_in_channel = 3 * self.history_len if self.input_rgb else self.input_c * self.history_len

    def init_model(self, input_c=None, out_c=None):
        """
        can re-write this to get general models
        :return:
        """
        if self.input_rgb:
            return CNN(output_c=out_c if out_c is not None else self.n_act,
                       input_c=input_c if input_c is not None else self.model_in_channel,
                       hidden_size=self.hidden_size)
        return FC_BN(output_c=out_c if out_c is not None else self.n_act,
                     input_c=input_c if input_c is not None else self.model_in_channel,
                     hidden_size=self.hidden_size,
                     lstm=self.lstm, n_layers=self.n_layers)

    def forward(self, obs_tensor):
        """
        process obs and feed obs to models
        """
        raise NotImplementedError

    def act(self, state):
        """
        procedure to select env-recognizable action

        also give methods for saving trajectories if training = True
        """
        raise NotImplementedError

    def record(self, reward, obs_tensor, action=None, finished=False):
        """
        reward, obs_tensor, action = [r, next_s, next_act] for the purpose of agent's 'sensor'
        initial state -> set r = 0
        add trajectory as [s, a, r, next_s, ..., r, final_s]
        :param reward: int
        :param obs_tensor: torch tensor without batch_dim
        :param action: int? or whatever you like
        :param finished: It's Final state ?
        """
        if self.training:
            if self.trajectory:  # r + gamma * max[Q(s', a')]
                self.trajectory += [reward, obs_tensor.squeeze(0)]
                if not finished:
                    self.trajectory += [action]
            else:  # initial state, the reward is None
                self.trajectory = [obs_tensor.squeeze(0), action]

    def backup(self):
        """
        process trajectories and add to replay buffer
        """
        raise NotImplementedError

    def train(self):
        """
        perform training process to update the model
        """
        raise NotImplementedError

    def load_ckp(self, ckp, training=False):
        """
        load models from ckp, usually for testing
        """
        raise NotImplementedError

class Actor(threading.Thread):
    """
    used for async-play
    """

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
