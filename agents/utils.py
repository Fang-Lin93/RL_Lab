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
