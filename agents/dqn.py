import random

import torch
from loguru import logger
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch import FloatTensor, LongTensor
from .basic import ReplayBuffer, BaseAgent
from .utils import process_image_obs, process_vec_obs

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class Buffer(ReplayBuffer):

    def sample(self, batch_size):
        return random.sample(self.data, batch_size)


class DQNAgent(BaseAgent):
    """
    TD: TD-target
    MC: MC-target

    Policy_model: newer ongoing model
    Target_model: older mostly fixed model
    """

    def __init__(self, eps_greedy: float = -1, target_type='TD', **kwargs):
        super().__init__(name='dqn', **kwargs)

        self.eps_greedy = eps_greedy
        self.target_type = target_type
        self.rb = Buffer(capacity=kwargs.get('buffer_size', 10000))

        self.policy_model = self.init_model().to(device)

        logger.info(f'Num paras={self.policy_model.num_paras()}')
        self.target_model = None

        if self.training:
            self.target_model = self.init_model().to(device)
            self.target_model.eval()
        self.sync_model()

    def act(self, state: dict):

        obs_, la, r, finished = state['obs'], state['la'], state['reward'], state['done']
        assert len(la) == self.n_act

        obs_tensor = process_image_obs(obs_, self.history_len) if self.input_rgb else \
            process_vec_obs(obs_, self.history_len, self.input_c, self.disable_byte_norm, self.lstm)

        action = None
        if not finished:
            if self.eps_greedy > 0 and random.random() < self.eps_greedy:
                action = random.choice(la)
            else:
                action = self.forward(obs_tensor)

        # save trajectories
        if self.training:
            self.record(r, obs_tensor, action, finished)

        return action

    def forward(self, obs_tensor):
        self.policy_model.eval()
        with torch.no_grad():
            pred_q = self.policy_model(obs_tensor.to(device))
            logger.debug(pred_q)
            return pred_q.view(-1).argmax().item()  # max part of Q-target

    def backup(self, final_payoff: float = 0):
        """
        If Q-MC, using MC-target, otherwise using TD target

        if LSTM -> a trajectory = a batch
        I can update the paras in the final stage (use total loss)
        """
        if self.trajectory:
            v = final_payoff
            for i in range(len(self.trajectory) - 2, 0, -3):
                if self.target_type == 'MC':  # simply ignore immediate rewards
                    self.trajectory[i] = v
                    v *= self.gamma
                # o, a, r, next_o
                self.rb.add((self.trajectory[i - 2],
                             self.trajectory[i - 1],
                             self.trajectory[i],
                             self.trajectory[i + 1]))

        self.trajectory = []
        return

    def train(self):
        """
        train POLICY model !!!
        """
        # self.process_trajectory()

        if len(self.rb) < self.batch_size:
            return

        self.policy_model.train()

        opt = torch.optim.RMSprop(self.policy_model.parameters(), lr=self.lr, eps=self.eps)

        logger.debug(
            f'====== Train Q net using {self.target_type} target (obs={len(self.rb)}) ======')

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

    def load_ckp(self, ckp, training=False):
        if not training:
            try:
                self.policy_model.load_state_dict(torch.load(f'checkpoints/{ckp}/target.pth', map_location='cpu'))
                self.policy_model.eval()
            except Exception as exp:
                raise ValueError(f'{exp}')
        else:
            try:
                self.policy_model.load_state_dict(torch.load(f'checkpoints/{ckp}/policy.pth', map_location='cpu'))
                self.target_model.load_state_dict(torch.load(f'checkpoints/{ckp}/target.pth', map_location='cpu'))
                self.policy_model.eval()
            except Exception as exp:
                raise ValueError(f'{exp}')


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
    agent.backup()
    history = [obs]
    # traj = [agent.process_vec_obs(history)]
    state_dict = {
        'obs': history,
        'la': list(range(env.action_space.n)),
        'reward': None,
        'done': False,
    }
    t, score, frame = 0, 0, 0
    act = None
    while True:
        env.render()
        if frame <= 0:
            act = agent.act(state_dict)
            frame = frame_freq - 1
            # traj.append(action)
        obs, reward, done, info = env.act(act)
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

        logger.debug(f'Action={act}, R_t+1={reward}')
        t += 1
        frame -= 1
        score += reward
        if done or t > 1000:
            state_dict['done'] = True
            agent.act(state_dict)
            logger.info(f"Episode finished after {t} time steps, total reward={score}")
            break

    if target == 'TD':
        agent.backup(final_payoff=reward)
    if target == 'MC':
        agent.backup(final_payoff=score)
