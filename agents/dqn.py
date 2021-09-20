import random
from abc import ABC

import torch
from loguru import logger
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch import nn, FloatTensor, LongTensor

"""
it uses LSTM to extract hidden states
"""
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class ReplayBuffer(object):

    def __init__(self, capacity: int = 10000, only_once=False):
        self.capacity = capacity
        self.only_once = only_once
        self.data = []

    def add(self, o, a, r, n_o):  # (s, a, r, ns) or (o, h, c, a, r, no, nh, nc) or (o, h, c, a, r)
        """Save a transition"""
        self.data.append((o, a, r, n_o))
        if len(self.data) > self.capacity:
            self.data.pop(0)

    def sample(self, batch_size):
        if self.only_once:
            random.shuffle(self.data)
            sample = self.data[:batch_size]
            self.data = self.data[batch_size:]
            return sample
        return random.sample(self.data, batch_size)

    def cla(self):
        self.data = []

    def is_full(self):
        return len(self.data) >= self.capacity

    def get_all_trajectories(self):
        return self.data

    def dataloader(self, batch_size):
        random.shuffle(self.data)
        return (self.data[i:i + batch_size] for i in range(0, len(self), batch_size))

    def __len__(self):
        return len(self.data)


class FC_Q(nn.Module):
    """
    (N, L, C) for LSTM
    otherwise: (N, L*C)
    with h_t <- (o_t, h_t-1)
    """

    def __init__(self, n_act: int, input_c: int, hidden_size: int = 32, n_layers: int = 6, lstm=False):
        super(FC_Q, self).__init__()
        self.n_act = n_act
        self.input_c = input_c
        self.hidden_size = hidden_size
        self.lstm = lstm
        self.rnn = None

        fc_in = input_c
        if lstm:
            self.rnn = nn.LSTM(input_c, hidden_size, batch_first=True)
            fc_in = hidden_size + input_c

        fc_layers = [i for _ in range(n_layers) for i in
                     (nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())]

        self.fc = nn.Sequential(nn.Linear(fc_in, hidden_size),
                                nn.ReLU(),
                                *fc_layers,
                                nn.Linear(hidden_size, n_act)
                                )

    def forward(self, obs_):
        """
        lstm -> (N=1, L, input_c)
        otherwise -> (N=1, L*input_c)
        """

        if self.lstm:
            lstm_out, (_, _) = self.rnn(obs_)
            x = torch.cat([obs_[:, -1, :], lstm_out[:, -1, :]], dim=-1)
            return self.fc(x).view(-1, self.n_act)

        return self.fc(obs_.view(-1, self.input_c)).view(-1, self.n_act)

    def save_model(self, model_file='v1', path='models/'):
        torch.save(self.state_dict(), f'{path}dqn_{model_file}.pth')

    def load_model(self, model_file='v0', path='models/'):
        self.load_state_dict(
            torch.load(f'{path}dqn_{model_file}.pth', map_location=torch.device('cpu')))

    def num_paras(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNN_Q(nn.Module):
    """
    (N, L*C, H, W)
    with h_t <- (o_t, h_t-1)

    use half -> (210, 105) -> (105, 80)
    """

    def __init__(self, n_act: int, input_c: int, height: int = 105, width: int = 80, hidden_size: int = 32):
        super(CNN_Q, self).__init__()
        self.input_c, self.height, self.width = input_c, height, width
        self.n_act = n_act

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=16, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
        )
        # self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        h = self.s_cv2d(self.s_cv2d(self.s_cv2d(height)))
        w = self.s_cv2d(self.s_cv2d(self.s_cv2d(width)))
        self.fc = nn.Sequential(
            nn.Linear(16*w*h, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_act)
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
        return self.fc(obs_).view(-1, self.n_act)
        # N, L, C, H, W = obs_.size()
        # obs_ = self.cnn(obs_.view(-1, C, H, W))
        # lstm_out, hidden_s = self.rnn(obs_.view(N, L, -1))
        # return self.fc(lstm_out[:, -1, :]).view(-1, self.n_act)

    def save_model(self, model_file='v1', path='models/'):
        torch.save(self.state_dict(), f'{path}dqn_{model_file}.pth')

    def load_model(self, model_file='v0', path='models/'):
        self.load_state_dict(
            torch.load(f'{path}dqn_{model_file}.pth', map_location=torch.device('cpu')))

    def num_paras(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DQNAgent(object):
    """
    TD: TD-target
    MC: MC-target

    Policy_model: newer ongoing model
    Target_model: older mostly fixed model
    """

    def __init__(self, n_act: int, model_file='v0', pre_train=False, eps_greedy: float = -1, training=False,
                 target_type: str = 'TD', input_rgb: bool = False, input_c: int = None, **kwargs):
        self.name = 'dqn'
        self.a_act = n_act
        self.eps_greedy = eps_greedy

        self.input_rgb = input_rgb
        self.input_c = input_c
        self.hidden_size = kwargs.get('hidden_size', 32)
        self.target_type = target_type
        self.rb = ReplayBuffer(capacity=kwargs.get('buffer_size', 10000))

        self.batch_size = kwargs.get('batch_size', 128)
        self.gamma = kwargs.get('gamma', 0.9)
        self.max_grad_norm = kwargs.get('max_grad_norm', 10)
        self.max_grad_value = kwargs.get('max_grad_value', 1)
        self.lr = kwargs.get('lr', 0.0001)
        self.eps = kwargs.get('eps', 1e-5)
        self.history_len = kwargs.get('history_len', 5)
        self.disable_byte_norm = kwargs.get('disable_byte_norm', False)
        self.n_layers = kwargs.get('n_layers', 6)
        self.lstm = kwargs.get('lstm', False)
        self.game = kwargs.get('game', 'game')

        if self.lstm:
            model_in_channel = 3 if self.input_rgb else input_c
        else:
            model_in_channel = 3 * self.history_len if self.input_rgb else input_c * self.history_len

        self.policy_model = CNN_Q(n_act, input_c=model_in_channel, hidden_size=self.hidden_size).to(device) \
            if input_rgb else FC_Q(n_act=n_act, input_c=model_in_channel, hidden_size=self.hidden_size,
                                   lstm=self.lstm, n_layers=self.n_layers).to(device)
        logger.info(f'Num paras={self.policy_model.num_paras()}')
        self.target_model = None
        self.training = training

        if training:
            self.target_model = CNN_Q(n_act, input_c=model_in_channel, hidden_size=self.hidden_size).to(device) \
                if input_rgb else FC_Q(n_act=n_act, input_c=model_in_channel, hidden_size=self.hidden_size,
                                       lstm=self.lstm, n_layers=self.n_layers).to(device)
            self.target_model.eval()

        try:
            self.target_model.load_state_dict(
                torch.load(f'models/dqn_{self.game}_v0.pth', map_location=torch.device('cpu')))
            logger.info(f'Successfully loaded checkpoint = models/dqn_{self.game}_v0.pth')

        except Exception as ex:
            print(ex)

        self.sync_model()

        # self.hidden_s = None
        self.trajectory = []  # (s, a, r, s') or (o, h, c, a ,r ,n_o ,n_h, n_c) // (o, a, r, no)

        if pre_train:
            self.policy_model.load_model(model_file)

    def reset(self):
        self.process_trajectory()
        self.trajectory = []
        # self.hidden_s = None

    def step(self, state: dict):
        obs_, la, r = state['obs'], state['la'], state['reward']
        o, a = self.act(obs_, la)

        # save trajectories
        if self.training:
            if self.trajectory:  # r + gamma * max[Q(s', a')]
                # self.trajectory += [r, o.squeeze(0), *(_.view(-1) for _ in self.hidden_s)]
                self.trajectory += [r, o.squeeze(0)]
                if not state['done']:
                    self.trajectory += [a]
            else:  # initial
                # self.trajectory = [o.squeeze(0), *(_.view(-1) for _ in self.hidden_s), a]
                self.trajectory = [o.squeeze(0), a]
        return a

    def act(self, obs_: list, la: list):
        assert len(la) == self.a_act
        # eps-greedy
        obs_tensor = self.process_image_obs(obs_) if self.input_rgb else self.process_vec_obs(obs_)

        self.policy_model.eval()
        with torch.no_grad():
            pred_q = self.policy_model(obs_tensor.to(device))
            logger.debug(pred_q)

            if self.eps_greedy > 0 and random.random() < self.eps_greedy:
                action_ = random.choice(la)
            else:
                action_ = pred_q.view(-1).argmax().item()

            return obs_tensor.cpu(), action_  # max part of Q-target

    def process_trajectory(self, final_payoff: float = 0):
        """
        If Q-MC, using MC-target, otherwise using TD target

        if LSTM -> a trajectory = a batch
        I can update the paras in the final stage (use total loss)
        """
        if self.trajectory:
            v = final_payoff
            # o, a, r, n_o = [], [], [], []
            for i in range(len(self.trajectory) - 2, 0, -3):
                if self.target_type == 'MC':  # simply ignore immediate rewards
                    self.trajectory[i] = v
                    v *= self.gamma

                self.rb.add(o=self.trajectory[i - 2],
                            a=self.trajectory[i - 1],
                            r=self.trajectory[i],
                            n_o=self.trajectory[i + 1])  # o, n_o -> torch.stack

        return

    def train_loop(self):
        """
        train POLICY model !!!
        """
        # self.process_trajectory()

        if len(self.rb) < self.batch_size:
            return

        self.policy_model.train()

        opt = torch.optim.RMSprop(self.policy_model.parameters(), lr=self.lr, eps=self.eps)

        logger.debug(  # self.rb.sample()
            f'====== Train Q net using {self.target_type} target (obs={len(self.rb)}) episodes ======')

        sample = self.rb.sample(self.batch_size)
        o, a, r, n_o = zip(*sample)
        o, a, r, n_o = torch.stack(o), LongTensor(a), FloatTensor(r), torch.stack(n_o)

        if self.target_type == 'TD':
            # TD target
            with torch.no_grad():
                r += self.gamma * self.target_model(n_o.to(device)).max(dim=1).values.detach().cpu()

        q_ = self.policy_model(o.to(device))  # policy model on the current state!

        opt.zero_grad()

        loss = F.smooth_l1_loss(q_[range(q_.size(0)), a.to(device)], r.to(device))
        loss.backward()

        clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
        clip_grad_value_(self.policy_model.parameters(), self.max_grad_value)
        opt.step()

        logger.debug(f'Loss = {loss.item():.6f}')
        self.policy_model.eval()
        return loss.item()

    def sync_model(self):
        if not self.target_model:
            return
        self.target_model.load_state_dict(self.policy_model.state_dict())

    def process_image_obs(self, obs_: list) -> torch.FloatTensor:
        """
        normalize RGB
        give size of (1, L, C, H, W)
        pad zeros at the beginning
        it contains most recent self.history_len frames
        """
        assert len(obs_) <= self.history_len

        obs_ = [torch.FloatTensor(o_.transpose(2, 0, 1)) for o_ in obs_]
        obs_tensor = torch.zeros((self.history_len * 3, 210, 160))
        obs_tensor[-len(obs_) * 3:] = torch.cat(obs_)
        pool = nn.AvgPool2d(kernel_size=(2, 2))  # to make it smaller
        return pool(obs_tensor) / 255.
        # return obs_tensor.unsqueeze(0) / 255.

    def process_vec_obs(self, obs_: list):
        """
        :param obs_:
        :return: size (L, input_c)
        """
        assert len(obs_) <= self.history_len
        obs_tensor = torch.zeros((self.history_len, self.input_c))
        obs_tensor[-len(obs_):] = torch.FloatTensor(obs_)
        if self.lstm:
            obs_tensor = obs_tensor.unsqueeze(0)

        if self.disable_byte_norm:
            return obs_tensor
        return obs_tensor / 255.

        # if self.disable_byte_norm:
        #     return obs_tensor.unsqueeze(0)
        # return obs_tensor.unsqueeze(0)/255.


if __name__ == '__main__':
    import gym

    frame_freq = 1
    history_len = 5

    env = gym.make('Breakout-v0')  # 'SpaceInvaders-v0'
    target = 'TD'
    # env = gym.make('Breakout-v0')
    agent = DQNAgent(n_act=env.action_space.n,
                     training=True,
                     eps_greedy=0.1,
                     target_type=target,
                     input_rgb=True,
                     history_len=history_len,
                     input_c=env.observation_space.shape[0],
                     lstm=False)

    obs = env.reset()
    agent.reset()
    history = [obs]
    # traj = [agent.process_vec_obs(history)]
    state_dict = {
        'obs': history,
        'la': list(range(env.action_space.n)),
        'reward': None,
        'done': False,
    }
    t, score, frame = 0, 0, 0
    action = None
    while True:
        env.render()
        if frame <= 0:
            action = agent.step(state_dict)
            frame = frame_freq - 1
            # traj.append(action)
        obs, reward, done, info = env.step(action)
        history.append(obs)
        if len(history) > history_len:
            history.pop(0)

        state_dict = {
            'obs': history,
            'la': list(range(env.action_space.n)),
            'reward': reward,
            'done': False
        }
        # traj.append(reward)
        # traj.append(agent.process_vec_obs(history))

        logger.debug(f'Action={action}, R_t+1={reward}')
        t += 1
        frame -= 1
        score += reward
        if done or t > 1000:
            state_dict['done'] = True
            agent.step(state_dict)
            logger.info(f"Episode finished after {t} time steps, total reward={score}")
            break

    if target == 'TD':
        agent.process_trajectory(final_payoff=reward)
    if target == 'MC':
        agent.process_trajectory(final_payoff=score)
