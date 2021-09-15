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

    def __init__(self, capacity: int = 10000, only_once=True):
        self.capacity = capacity
        self.one_shot = only_once
        self.data = []

    def add(self, o, a, r, n_o):  # (s, a, r, ns) or (o, h, c, a, r, no, nh, nc) or (o, h, c, a, r)
        """Save a transition"""
        self.data.append((o, a, r, n_o))
        if len(self.data) > self.capacity:
            self.data.pop(0)

    def sample(self, batch_size):
        if self.one_shot:
            random.shuffle(self.data)
            sample = self.data[:batch_size]
            self.data = self.data[batch_size:]
            return sample
        return random.sample(self.data, batch_size)

    def cla(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def is_full(self):
        return len(self.data) >= self.capacity

    def get_all_trajectories(self):
        return self.data

    def dataloader(self, batch_size):
        random.shuffle(self.data)
        return (self.data[i:i + batch_size] for i in range(0, len(self), batch_size))


class FC_Q(nn.Module):
    """
    (N, L, C)
    with h_t <- (o_t, h_t-1)
    """

    def __init__(self, n_act: int, input_c: int, hidden_size: int = 256):
        super(FC_Q, self).__init__()
        self.n_act = n_act
        self.input_c = input_c

        self.rnn = nn.LSTM(input_c, hidden_size, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_act)
        )

    def forward(self, obs_):
        """
        x of size (3, 210, 160) , if (210, 160, 3), try using tensor.permute(2, 0, 1)
        lens_idx = [0, 1,5, 2,...] is the index of outputs for different trajectories (= length_seq - 1)

        single playing: N=1, L = len(history_frames), C, H, W = (3, 210, 160), lens_idx = [len(history_frames)-1]
        """
        lstm_out, (_, _) = self.rnn(obs_)

        return self.fc(lstm_out[:, -1, :]).view(-1, self.n_act)

    def save_model(self, model_file='v1', path='models/'):
        torch.save(self.state_dict(), f'{path}dqn_fc_{model_file}.pth')

    def load_model(self, model_file='v0', path='models/'):
        self.load_state_dict(
            torch.load(f'{path}dqn_fc_{model_file}.pth', map_location=torch.device('cpu')))

    def num_paras(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNN_Q(nn.Module):
    """
    (N, L, C, H, W)
    with h_t <- (o_t, h_t-1)
    """

    def __init__(self, n_act: int, input_size: tuple = (3, 210, 160), hidden_size: int = 128):
        super(CNN_Q, self).__init__()
        self.in_c, self.height, self.width = input_size
        self.n_act = n_act

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(self.height, self.width)),  # global max pooling
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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_act)
        )

    def forward(self, obs_):
        """
        x of size (3, 210, 160) , if (210, 160, 3), try using tensor.permute(2, 0, 1)
        lens_idx = [0, 1,5, 2,...] is the index of outputs for different trajectories (= length_seq - 1)

        single playing: N=1, L = len(history_frames), C, H, W = (3, 210, 160), lens_idx = [len(history_frames)-1]
        """
        N, L, C, H, W = obs_.size()

        obs_ = self.cnn(obs_.view(-1, C, H, W))
        lstm_out, hidden_s = self.rnn(obs_.view(N, L, -1))

        return self.fc(lstm_out[:, -1, :]).view(-1, self.n_act)

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
                 target_type: str = 'TD', input_rgb: bool = False, input_c: int = None, **kwargs):
        self.name = 'dqn'
        self.a_act = n_act
        self.eps_greedy = eps_greedy

        self.input_rgb = input_rgb
        self.input_c = input_c
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.policy_model = CNN_Q(n_act, hidden_size=self.hidden_size).to(device) \
            if input_rgb else FC_Q(n_act=n_act, input_c=input_c, hidden_size=self.hidden_size).to(device)
        self.target_model = None
        self.training = training

        if training:
            self.target_model = CNN_Q(n_act, hidden_size=self.hidden_size).to(device) \
                if input_rgb else FC_Q(n_act=n_act, input_c=input_c, hidden_size=self.hidden_size).to(device)
        self.sync_model()

        self.target_type = target_type
        self.rb = ReplayBuffer(capacity=kwargs.get('buffer_size', 10000))

        self.batch_size = kwargs.get('batch_size', 64)
        self.gamma = kwargs.get('gamma', 0.9)
        self.max_grad_norm = kwargs.get('max_grad_norm', 40)
        self.lr = kwargs.get('lr', 0.0001)
        self.eps = kwargs.get('eps', 1e-5)
        self.history_len = kwargs.get('history_len', 5)

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

                    # r = self.trajectory[i]
                    # self.trajectory[i] = r + self.gamma * v  # reward = current + future
                    # v = r + self.gamma * v

                self.rb.add(o=self.trajectory[i - 2],
                            a=self.trajectory[i - 1],
                            r=self.trajectory[i],
                            n_o=self.trajectory[i + 1])  # o, n_o -> torch.stack

        return

    def train_loop(self):
        """
        train target model
        """
        self.process_trajectory()

        if len(self.rb) < self.batch_size:
            return

        self.target_model.train()

        opt = torch.optim.RMSprop(self.target_model.parameters(), lr=self.lr, eps=self.eps)

        logger.info(  # self.rb.sample()
            f'====== Train Q net using {self.target_type} target (obs={len(self.rb)}) episodes ======')

        sample = self.rb.sample(self.batch_size)
        o, a, r, n_o = zip(*sample)
        o, a, r, n_o = torch.stack(o), LongTensor(a), FloatTensor(r), torch.stack(n_o)

        q_ = self.target_model(n_o.to(device))
        if self.target_type == 'TD':
            # get the td target first
            r += q_.max(dim=1).values.detach().cpu()

        opt.zero_grad()

        loss = F.mse_loss(q_[range(q_.size(0)), a.to(device)], r.to(device))
        loss.backward()
        clip_grad_norm_(self.target_model.parameters(), self.max_grad_norm)
        opt.step()

        logger.info(f'Loss = {loss.item():.6f}')
        self.target_model.eval()

    def sync_model(self):
        if not self.target_model:
            return
        self.policy_model.load_state_dict(self.target_model.state_dict())

    def process_image_obs(self, obs_: list) -> torch.FloatTensor:
        """
        normalize RGB
        give size of (1, L, C, H, W)
        pad zeros at the beginning
        it contains most recent self.history_len frames
        """
        assert len(obs_) <= self.history_len

        obs_ = [torch.FloatTensor(o_.transpose(2, 0, 1)) for o_ in obs_]
        obs_tensor = torch.zeros((self.history_len, *obs_[0].shape))
        obs_tensor[-len(obs_):] = torch.stack(obs_)
        return obs_tensor.unsqueeze(0) / 255.

    def process_vec_obs(self, obs_: list) -> torch.FloatTensor:
        """
        :param obs_:
        :return: size (1, L, input_c)
        """
        assert len(obs_) <= self.history_len
        obs_tensor = torch.zeros((self.history_len, self.input_c))
        obs_tensor[-len(obs_):] = torch.FloatTensor(obs_)
        return obs_tensor.unsqueeze(0)/255.


if __name__ == '__main__':
    import gym

    frame_freq = 3
    history_len = 10

    env = gym.make('SpaceInvaders-ram-v0')  # 'SpaceInvaders-v0'
    target = 'TD'
    # env = gym.make('Breakout-v0')
    agent = DQNAgent(n_act=env.action_space.n,
                     training=True,
                     eps_greedy=0.1,
                     target_type=target,
                     input_rgb=False,
                     history_len=history_len,
                     input_c=env.observation_space.shape[0])

    obs = env.reset()
    agent.reset()
    history = [obs]
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
        if frame == 0:
            action = agent.step(state_dict)
            frame = frame_freq - 1

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
