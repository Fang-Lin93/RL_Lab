import random
import torch
from torch import nn

"""
This transcript aims to store functions used for normalizing inputs
"""


def process_image_obs(obs_: list, history_len: int) -> torch.FloatTensor:
    """
    normalize RGB of Atari frames
    give size of (1, L, C, H, W)
    pad zeros at the beginning
    it contains most recent self.history_len frames
    """
    assert len(obs_) <= history_len

    obs_ = [torch.FloatTensor(o_.transpose(2, 0, 1)) for o_ in obs_]
    obs_tensor = torch.zeros((history_len * 3, 210, 160))
    obs_tensor[-len(obs_) * 3:] = torch.cat(obs_)
    pool = nn.AvgPool2d(kernel_size=(2, 2))
    return pool(obs_tensor) / 255.


def process_vec_obs(obs_: list, history_len: int, input_c: int, disable_byte_norm=False, lstm=False):
    """
    obs_ = [vec, vec, ...] (past -> future)
    """
    assert len(obs_) <= history_len
    obs_tensor = torch.zeros((history_len, input_c))
    obs_tensor[-len(obs_):] = torch.FloatTensor(obs_)
    if lstm:
        obs_tensor = obs_tensor.unsqueeze(0)
    if disable_byte_norm:
        return obs_tensor
    return obs_tensor / 255.


class ReplayBuffer(object):

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.data = []

    def add(self, o, a, r, n_o):  # (s, a, r, ns) or (o, h, c, a, r, no, nh, nc) or (o, h, c, a, r)
        """Save a transition"""
        self.data.append((o, a, r, n_o))
        if len(self.data) > self.capacity:
            self.data.pop(0)

    def sample(self, batch_size):
        return random.sample(self.data, batch_size)

    def cla(self):
        self.data = []

    def is_full(self):
        return len(self.data) >= self.capacity

    # def get_all_trajectories(self):
    #     return self.data

    # def dataloader(self, batch_size):
    #     random.shuffle(self.data)
    #     return (self.data[i:i + batch_size] for i in range(0, len(self), batch_size))

    def __len__(self):
        return len(self.data)
