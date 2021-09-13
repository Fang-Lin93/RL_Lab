
import random
import torch
from torch.nn import functional as F
from torch import nn


class CNN_Act(nn.Module):
    """
    (N, C, H, W)
    """
    def __init__(self, n_act: int, input_size: tuple = (3, 210, 160), hidden_size: int = 128):
        super(CNN_Act, self).__init__()
        self.in_c, self.height, self.width = input_size
        self.n_act = n_act

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(self.height, self.width)),  # global max pooling
            nn.Flatten(),
        )
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_act)
        )

    def forward(self, obs, lens_idx: list = None, hidden_state=None):
        """
        x of size (3, 210, 160) , if (210, 160, 3), try using tensor.permute(2, 0, 1)
        lens_idx = [0, 1,5, 2,...] is the index of outputs for different trajectories (= length_seq - 1)

        single playing: N=1, L = len(history_frames), C, H, W = (3, 210, 160), lens_idx = [len(history_frames)-1]
        """
        N, L, C, H, W = obs.size()
        if lens_idx is None:
            lens_idx = [L-1]

        obs = self.cnn(obs.view(-1, C, H, W))
        lstm_out, hidden_s = self.rnn(obs.view(N, L, -1))

        return self.fc(lstm_out[range(N), lens_idx, :]), hidden_s

    def train_loop(self, data: tuple, device):
        """
        SL: x = (B, 88), y= (B, 2) here is [[-1, ,1], ...]
        RL: x = (B, 88), a_ = [0 or 1]*B, y_ = [payoff]*B,
        """
        x_, len_idx_, y_ = data
        x_, len_idx_, y_ = x_.to(device), len_idx_.to(device), y_.to(device)
        pred_y = self(x_, len_idx_)

        loss, acc = self.loss_function(pred_y, y_)
        return loss, acc / len(x_)

    @staticmethod
    def loss_function(pred_q, target_q):
        """
        MSE for q-value -> final_payoff
        """
        return F.mse_loss(pred_q, target_q), pred_q.round().eq(target_q).sum()

    def save_model(self, model_file='v1', path='models/'):
        torch.save(self.state_dict(), f'{path}dqn_{model_file}')

    def load_model(self, model_file='v0', path='models/'):
        self.load_state_dict(
            torch.load(f'{path}dqn_{model_file}', map_location=torch.device('cpu')))

    def num_paras(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DQNAgent(object):
    def __init__(self, n_act: int, model_file='v0', pre_train=False, eps_greedy=-1):
        self.name = 'dqn'
        self.a_act = n_act
        self.eps_greedy = eps_greedy
        self.model = CNN_Act(n_act)
        self.hidden_s = None
        if pre_train:
            self.model.load_model(model_file)

    def step(self, obs: list, la: list):
        assert len(la) == self.a_act
        # eps-greedy
        if self.eps_greedy > 0 and random.random() < self.eps_greedy:
            return random.choice(la)

        self.model.eval()
        obs_tensor = self.process_obs(obs)
        with torch.no_grad():
            pred_q, self.hidden_s = self.model(obs_tensor, hidden_state=self.hidden_s)
            print(pred_q)
            return pred_q.view(-1).argmax().item()

    @staticmethod
    def process_obs(obs: list) -> torch.FloatTensor:
        """
        normalize RGB
        """
        obs_tensor = torch.FloatTensor([o_.transpose(2, 0, 1) for o_ in obs]).unsqueeze(0)

        return obs_tensor/255.


if __name__ == '__main__':
    self = CNN_Act(n_act=6)
    x = torch.rand(1, 10, 3, 210, 160)
    print(self(x))












