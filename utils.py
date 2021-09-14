import torch


def process_obs(obs_: list, length: int) -> torch.FloatTensor:
    """
    normalize RGB
    give size of (L, C, H, W)
    pad zeros at the beginning
    """
    obs_ = [torch.FloatTensor(o_.transpose(2, 0, 1)) for o_ in obs_]
    obs_tensor = torch.zeros((length, *obs_[0].shape))
    obs_tensor[-len(obs_):] = torch.stack(obs_)

    return obs_tensor / 255.




