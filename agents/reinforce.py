import random
import torch
from loguru import logger
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch import FloatTensor, LongTensor
from .basic import ReplayBuffer, Agent
from .utils import process_image_obs, process_vec_obs
from torch.distributions import Categorical

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class Buffer(ReplayBuffer):

    def sample(self, batch_size):
        random.shuffle(self.data)
        return (self.data[i:i + batch_size] for i in range(0, len(self), batch_size))


class REINFO_Agent(Agent):
    """
    policy gradient with Q-value estimated by MC return (with base-line)
    """

    def __init__(self, **kwargs):
        super().__init__(name='dqn', **kwargs)

        self.rb = Buffer(capacity=kwargs.get('buffer_size', 10000))
        self.policy_model = self.init_model().to(device)

        self.base_line = self.init_model(out_c=1).to(device)

        logger.info(f'Num paras(policy)={self.policy_model.num_paras()}')

        self.policy_model.eval()

        if self.training:
            self.base_line = self.init_model(out_c=1)
            self.base_line.train()
            logger.info(f'Num paras(baseline)={self.base_line.num_paras()}')

    def act(self, state: dict):

        obs_, la, r, finished = state['obs'], state['la'], state['reward'], state['done']
        assert len(la) == self.n_act

        obs_tensor = process_image_obs(obs_, self.history_len) if self.input_rgb else \
            process_vec_obs(obs_, self.history_len, self.input_c, self.disable_byte_norm, self.lstm)

        action = None
        if not finished:
            action = self.forward(obs_tensor)

        # save trajectories
        if self.training:
            self.record(r, obs_tensor, action, finished)

        return action

    def forward(self, obs_tensor):
        self.policy_model.eval()
        with torch.no_grad():
            prob = Categorical(logits=self.policy_model(obs_tensor.to(device)).view(-1))
            return prob.sample().item()  # max part of Q-target

    def backup(self):
        """
        If Q-MC, using MC-target, otherwise using TD target

        if LSTM -> a trajectory = a batch
        I can update the paras in the final stage (use total loss)
        """
        if self.trajectory:
            v = 0
            for i in range(len(self.trajectory) - 2, 0, -3):
                v += self.trajectory[i]  # immediate reward
                # o, a, r, next_o
                self.rb.add((self.trajectory[i - 2],
                             self.trajectory[i - 1],
                             self.trajectory[i],
                             self.gamma ** (i // 3)))  # discounted to the starting point

                v *= self.gamma

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

        opt_policy = torch.optim.RMSprop(self.policy_model.parameters(), lr=self.lr, eps=self.eps)
        opt_base = torch.optim.RMSprop(self.base_line.parameters(), lr=self.lr, eps=self.eps)

        logger.debug(f'====== Train REINFORCE (obs={len(self.rb)}) ======')

        t_v_loss, t_p_loss, counts = 0, 0, 0

        for batch in self.rb.sample(self.batch_size):

            opt_policy.zero_grad()
            opt_base.zero_grad()
            o, a, r, d = [], [], [], []
            for item in batch:
                o.append(item[0])
                a.append(item[1])
                r.append(item[2])
                d.append(item[3])

            o, a, r, d = torch.stack(o), LongTensor(a), FloatTensor(r), FloatTensor(d)
            # train value baseline
            self.base_line.train()
            pred_v = self.base_line(o.to(device)).view(-1)
            v_loss = F.smooth_l1_loss(pred_v.to(device), r.to(device))
            v_loss.backward()
            clip_grad_norm_(self.base_line.parameters(), self.max_grad_norm)
            opt_base.step()

            # train policies
            self.base_line.eval()

            critic = (r - self.base_line(o.to(device)).view(-1).detach()) * d
            p_loss = (critic * self.policy_model(o).softmax(dim=-1)[range(o.size(0)), a].log()).mean()
            p_loss.backward()
            clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            opt_policy.step()

            t_v_loss += v_loss.item()*len(batch)
            t_p_loss += p_loss.item()*len(batch)
            counts += len(batch)
        self.rb.cla()
        t_v_loss /= counts
        t_p_loss /= counts
        logger.info(f'ValueLoss={t_v_loss}, PolicyLoss={t_p_loss}')
        return t_v_loss, t_p_loss


if __name__ == '__main__':
    import gym

    frame_freq = 1
    history_len = 5

    env = gym.make('CartPole-v0')  # 'SpaceInvaders-v0'
    target = 'TD'
    # env = gym.make('Breakout-v0')
    agent = REINFO_Agent(n_act=env.action_space.n,
                         history_len=history_len,
                         input_c=env.observation_space.shape[0])

    obs = env.reset()
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
        obs, reward, done, info = env.step(act)
        print(reward)
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

    agent.backup()

