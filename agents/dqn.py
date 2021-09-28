import random

import torch
from loguru import logger
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch import FloatTensor, LongTensor
from .utils import ReplayBuffer, process_image_obs, process_vec_obs
from .networks.Q import FC_BN, CNN
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


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

        self.policy_model = CNN(n_act, input_c=model_in_channel, hidden_size=self.hidden_size).to(device) \
            if input_rgb else FC_BN(n_act=n_act, input_c=model_in_channel, hidden_size=self.hidden_size,
                                    lstm=self.lstm, n_layers=self.n_layers).to(device)
        logger.info(f'Num paras={self.policy_model.num_paras()}')
        self.target_model = None
        self.training = training

        if training:
            self.target_model = CNN(n_act, input_c=model_in_channel, hidden_size=self.hidden_size).to(device) \
                if input_rgb else FC_BN(n_act=n_act, input_c=model_in_channel, hidden_size=self.hidden_size,
                                        lstm=self.lstm, n_layers=self.n_layers).to(device)
            self.target_model.eval()

        if self.target_model:
            try:
                self.target_model.load_state_dict(
                    torch.load(f'models/dqn_{self.game}_v0.pth', map_location=torch.device('cpu')))
                logger.info(f'Successfully loaded checkpoint = models/dqn_{self.game}_v0.pth')

            except Exception as ex:
                logger.debug(ex)

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
        obs_tensor = process_image_obs(obs_, self.history_len) if self.input_rgb else \
            process_vec_obs(obs_, self.history_len, self.input_c, self.disable_byte_norm, self.lstm)

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
        opt.step()

        logger.debug(f'Loss = {loss.item():.6f}')
        self.policy_model.eval()
        return loss.item()

    def sync_model(self):
        if not self.target_model:
            return
        self.target_model.load_state_dict(self.policy_model.state_dict())


if __name__ == '__main__':
    import gym

    frame_freq = 1
    history_len = 5

    env = gym.make('Breakout-ram-v4')  # 'SpaceInvaders-v0'
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
