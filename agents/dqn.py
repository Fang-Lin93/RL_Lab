import random
from abc import ABC

import torch
from loguru import logger
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch import nn, FloatTensor, LongTensor

"""
it uses LSTM to extract hidden states
"""
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class ReplayBuffer(object):

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.data = []

    def add(self, *args):  # (s, a, r, ns) or (o, h, c, a, r, no, nh, nc) or (o, h, c, a, r)
        """Save a transition"""
        self.data.append(args)
        if len(self.data) > self.capacity:
            self.data.pop(0)

    def sample(self, batch_size):
        random.shuffle(self.data)
        sample = self.data[:batch_size]
        self.data = self.data[batch_size:]
        return sample

    def __len__(self):
        return len(self.data)

    def __get_is_full(self):
        return len(self.data) >= self.capacity

    is_full = property(__get_is_full)


class CNN_Act(nn.Module):
    """
    (N, C, H, W)
    with h_t <- (o_t, h_t-1)
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

    def forward(self, obs_, hidden_state=None, lens_idx: list = None):
        """
        x of size (3, 210, 160) , if (210, 160, 3), try using tensor.permute(2, 0, 1)
        lens_idx = [0, 1,5, 2,...] is the index of outputs for different trajectories (= length_seq - 1)

        single playing: N=1, L = len(history_frames), C, H, W = (3, 210, 160), lens_idx = [len(history_frames)-1]
        """
        if lens_idx is None:
            lens_idx = [obs_.size(1) - 1] * obs_.size(0)

        lstm_out, hidden_s = self.update_obs(obs_, hidden_state)

        return self.fc(lstm_out[range(obs_.size(0)), lens_idx, :]), hidden_s

    def update_obs(self, obs_, hidden_state):
        N, L, C, H, W = obs_.size()

        obs_ = self.cnn(obs_.view(-1, C, H, W))
        lstm_out, hidden_s = self.rnn(obs_.view(N, L, -1), hidden_state)
        return lstm_out, hidden_s

    def train_loop(self, data: tuple, device_=device):
        o_, h_, c_, a_, r_ = data
        # o_, h_, c_, a_, r_, no_, nh_, oc_ = data
        # o_, h_, c_, a_, r_, no_, nh_, oc_ = o_.to(device_), h_.to(device_), c_.to(device_), a_.to(device_), \
        #                                     r_.to(device_), no_.to(device_), nh_.to(device_), oc_.to(device_)
        # h_, c_, nh_, oc_ = h_.unsqueeze(0), c_.unsqueeze(0), nh_.unsqueeze(0), oc_.unsqueeze(0)
        # h_, c_ = h_.unsqueeze(0), c_.unsqueeze(0)
        pred_q, _ = self(o_.to(device_), hidden_state=(h_.to(device_), c_.to(device_)))
        pred_q = pred_q[range(pred_q.size(0)), a_]
        loss = self.loss_function(pred_q, r_.to(device_))
        return loss

    @staticmethod
    def loss_function(pred_q, target_q):
        """
        MSE for q-value -> final_payoff
        """
        return F.mse_loss(pred_q, target_q)

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
    """

    def __init__(self, n_act: int, model_file='v0', pre_train=False, eps_greedy: float = -1, training=False,
                 target_type: str = 'TD', **kwargs):
        self.name = 'dqn'
        self.a_act = n_act
        self.eps_greedy = eps_greedy

        self.policy_model = CNN_Act(n_act).to(device)
        self.target_model = None
        self.training = training

        if training:
            self.target_model = CNN_Act(n_act).to(device)
        self.sync_model()

        self.target_type = target_type
        self.rb = ReplayBuffer(capacity=kwargs.get('buffer_size', 10000))

        self.batch_size = kwargs.get('batch_size', 256)
        self.gamma = kwargs.get('gamma', 0.9)
        self.max_grad_norm = kwargs.get('max_grad_norm', 40)

        self.hidden_s = None
        self.trajectory = []  # (s, a, r, s') or (o, h, c, a ,r ,n_o ,n_h, n_c)

        if pre_train:
            self.policy_model.load_model(model_file)

    def reset(self):
        self.hidden_s = None
        self.trajectory = []

    def step(self, state: dict):

        obs_, la, r = state['obs'], state['la'], state['reward']
        o, a, max_q = self.act(obs_, la)

        # save trajectories
        if self.training:
            if self.trajectory:  # r + gamma * max[Q(s', a')]
                self.trajectory += [r, o.squeeze(0), *(_.view(-1) for _ in self.hidden_s)]
                if not state['done']:
                    self.trajectory += [a]
            else:  # initial
                self.trajectory = [o.squeeze(0), *(_.view(-1) for _ in self.hidden_s), a]

        return a

    def act(self, obs_: list, la: list):
        assert len(la) == self.a_act
        # eps-greedy
        obs_tensor = self.process_obs(obs_)

        self.policy_model.eval()
        with torch.no_grad():
            pred_q, self.hidden_s = self.policy_model(obs_tensor.to(device), hidden_state=self.hidden_s)

            if self.eps_greedy > 0 and random.random() < self.eps_greedy:
                action_ = random.choice(la)
            else:
                action_ = pred_q.view(-1).argmax().item()

            return obs_tensor.cpu(), action_, pred_q.max().cpu().item()  # max part of Q-target

    def process_trajectory(self):
        """
        If Q-MC, using MC-target, otherwise using TD target
        """
        if self.trajectory:
            v = 0
            for i in range(len(self.trajectory) - 4, 0, -5):
                if self.target_type == 'MC':
                    r = self.trajectory[i]
                    self.trajectory[i] = r + self.gamma * v  # reward = current + future
                    v = r + self.gamma * v
                    self.rb.add(*self.trajectory[i - 4:i + 1])
                else:  # TD error
                    self.rb.add(*self.trajectory[i - 4:i + 4])

        self.reset()

    def sample_data(self) -> tuple:
        """
        mc: (o, h, c, a, r*)
        td: (o, h, c, a, r, no, nh, nc)
        """
        self.policy_model.eval()
        with torch.no_grad():
            sample = self.rb.sample(self.batch_size)
            if self.target_type == 'TD':
                o, h, c, a, r, no, nh, nc = zip(*sample)
                o, h, c = torch.stack(o), torch.stack(h).unsqueeze(0), torch.stack(c).unsqueeze(0)
                a, r = LongTensor(a), FloatTensor(r)
                no, nh, nc = torch.stack(no), torch.stack(nh).unsqueeze(0), torch.stack(nc).unsqueeze(0)
                pred_q, _ = self.policy_model(no.to(device), (nh.to(device), nc.to(device)))
                r += self.gamma*pred_q.max(dim=1).values.cpu()
                return o, h, c, a, r

            elif self.target_type == 'MC':
                o, h, c, a, r = zip(*sample)
                return torch.stack(o), torch.stack(h).unsqueeze(0), torch.stack(c).unsqueeze(0), \
                    LongTensor(a), FloatTensor(r)
            else:
                raise ValueError(f'{self.target_type} is not accepted, acceptable types are in ["MC", "TD"]')

    def train_loop(self):
        self.process_trajectory()

        if len(self.rb) < self.batch_size:
            return

        self.target_model.train()

        opt = torch.optim.RMSprop(self.target_model.parameters(), lr=0.001, eps=1e-5)

        data = self.sample_data()

        logger.info(f'====== Train Q net using {self.target_type} target ======')

        loss = self.target_model.train_loop(data, device_=device)
        opt.zero_grad()
        loss.backward()
        clip_grad_norm_(self.target_model.parameters(), self.max_grad_norm)
        opt.step()

        logger.info(f'Loss: {loss:.6f}')
        self.target_model.eval()

    def sync_model(self):
        if not self.target_model:
            return
        self.policy_model.load_state_dict(self.target_model.state_dict())

    @staticmethod
    def process_obs(obs: list) -> torch.FloatTensor:
        """
        normalize RGB
        """
        obs_tensor = torch.FloatTensor([o_.transpose(2, 0, 1) for o_ in obs]).unsqueeze(0)

        return obs_tensor / 255.


if __name__ == '__main__':
    import gym

    env = gym.make('SpaceInvaders-v0')
    agent = DQNAgent(n_act=6, training=True, eps_greedy=0.1)

    obs = env.reset()
    agent.reset()
    state_dict = {
        'obs': [obs],
        'la': list(range(env.action_space.n)),
        'reward': None,
        'done': False,
    }
    t, score, frame = 0, 0, 4
    action = None
    while True:
        env.render()
        if frame == 4:
            action = agent.step(state_dict)
            frame = 0

        obs, reward, done, info = env.step(action)
        state_dict = {
            'obs': [obs],
            'la': list(range(env.action_space.n)),
            'reward': reward,
            'done': False
        }

        logger.debug(f'Action={action}, R_t+1={reward}')
        t += 1
        frame += 1
        score += reward
        if done or t > 1000:
            state_dict['done'] = True
            agent.step(state_dict)
            logger.info(f"Episode finished after {t} time steps, total reward={score}")
            break
