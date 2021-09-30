

import torch
from torch import nn


class FC_BN(nn.Module):
    """
    (N, L, C) for LSTM
    otherwise: (N, L*C)
    with h_t <- (o_t, h_t-1)
    """

    def __init__(self, output_c: int, input_c: int, hidden_size: int = 32, n_layers: int = 6, lstm=False):
        super(FC_BN, self).__init__()
        self.output_c = output_c
        self.input_c = input_c
        self.hidden_size = hidden_size
        self.lstm = lstm
        self.rnn = None

        fc_in = input_c
        if lstm:
            self.rnn = nn.LSTM(input_c, hidden_size, batch_first=True)
            fc_in = hidden_size + input_c

        fc_layers = [i for _ in range(n_layers) for i in
                     (nn.Linear(hidden_size, hidden_size),
                      nn.BatchNorm1d(hidden_size),
                      nn.ReLU())]

        self.fc = nn.Sequential(nn.Linear(fc_in, hidden_size),
                                nn.ReLU(),
                                *fc_layers,
                                nn.Linear(hidden_size, output_c)
                                )

    def forward(self, obs_):
        """
        lstm -> (N=1, L, input_c)
        otherwise -> (N=1, L*input_c)
        """

        if self.lstm:
            lstm_out, (_, _) = self.rnn(obs_)
            x = torch.cat([obs_[:, -1, :], lstm_out[:, -1, :]], dim=-1)
            return self.fc(x).view(-1, self.output_c)

        return self.fc(obs_.view(-1, self.input_c)).view(-1, self.output_c)

    def save_model(self, model_file='v1', path='models'):
        torch.save(self.state_dict(), f'{path}/dqn_{model_file}.pth')

    def load_model(self, model_file='v0', path='models'):
        self.load_state_dict(
            torch.load(f'{path}/dqn_{model_file}.pth', map_location=torch.device('cpu')))

    def num_paras(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNN(nn.Module):
    """
    (N, L*C, H, W)
    with h_t <- (o_t, h_t-1)

    use half -> (210, 105) -> (105, 80)
    """

    def __init__(self, output_c: int, input_c: int, height: int = 105, width: int = 80, hidden_size: int = 32):
        super(CNN, self).__init__()
        self.input_c, self.height, self.width = input_c, height, width
        self.output_c = output_c

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=hidden_size, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Flatten(),
        )
        # self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        h = self.s_cv2d(self.s_cv2d(self.s_cv2d(height)))
        w = self.s_cv2d(self.s_cv2d(self.s_cv2d(width)))
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*w*h, 256),
            nn.ReLU(),
            nn.Linear(256, output_c)
        )

    @staticmethod
    def s_cv2d(size, kernel_size=4, stride=2, padding=0):
        return (size + 2*padding - kernel_size) // stride + 1

    def forward(self, obs_):
        """
        x of size (3, 210, 160) , if (210, 160, 3), try using tensor.permute(2, 0, 1)
        lens_idx = [0, 1,5, 2,...] is the index of outputs for different trajectories (= length_seq - 1)

        single playing: N=1, L = len(history_frames), C, H, W = (3, 210, 160), lens_idx = [len(history_frames)-1]
        """
        obs_ = self.cnn(obs_.view(-1, self.input_c, self.height, self.width))  # (N, L*C, H, W)
        return self.fc(obs_).view(-1, self.output_c)
        # N, L, C, H, W = obs_.size()
        # obs_ = self.cnn(obs_.view(-1, C, H, W))
        # lstm_out, hidden_s = self.rnn(obs_.view(N, L, -1))
        # return self.fc(lstm_out[:, -1, :]).view(-1, self.output_c)

    def save_model(self, model_file='v1', path='models'):
        torch.save(self.state_dict(), f'{path}/dqn_{model_file}.pth')

    def load_model(self, model_file='v0', path='models'):
        self.load_state_dict(
            torch.load(f'{path}/dqn_{model_file}.pth', map_location=torch.device('cpu')))

    def num_paras(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

