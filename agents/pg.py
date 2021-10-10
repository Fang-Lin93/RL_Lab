import random
import torch
from loguru import logger
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch import FloatTensor, LongTensor
from .basic import ReplayBuffer, BaseAgent
from .utils import process_image_obs, process_vec_obs
from torch.distributions import Categorical

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class Buffer(ReplayBuffer):

    def sample(self, batch_size):
        random.shuffle(self.data)
        return (self.collate_fn(self.data[i:i + batch_size]) for i in range(0, len(self), batch_size))

    @staticmethod
    def collate_fn(batch):
        o, a, r, n_o, d = [], [], [], [], []
        for item in batch:
            o.append(item[0])
            a.append(item[1])
            r.append(item[2])
            n_o.append(item[3])
            d.append(item[4])
        return torch.stack(o), LongTensor(a), FloatTensor(r), torch.stack(n_o), FloatTensor(d)


class PGAgent(BaseAgent):
    """
    policy gradient use critics...

    mc: REINFORCE, MC - baseline -> target=mc
    q: action-value function -> target=td
    adv: advantage = r + V(s') - V(s) -> target=td
    """
    critic_types = ['mc', 'q', 'adv']
    critic_targets = ['mc', 'td']

    def __init__(self, critic='mc', critic_target='mc', **kwargs):
        super().__init__(name='policy_gradient', **kwargs)

        self.policy_model = self.init_model().to(device)
        self.policy_model.eval()
        logger.info(f'Num paras(policy)={self.policy_model.num_paras()}')

        self.critic = critic
        self.critic_target = critic_target
        if self.training:
            if self.critic == 'mc':  # base_line V(s) function
                self.critic_model = self.init_model(out_c=1).to(device)
            elif self.critic == 'q':  # Q-func
                self.critic_model = self.init_model().to(device)
            elif self.critic == 'adv':  # state-value
                self.critic_model = self.init_model(out_c=1).to(device)
            else:
                raise ValueError(f'Critic should be one of the types from {self.critic_types}, given {critic}')
            logger.info(f'Num paras(critic model)={self.critic_model.num_paras()}')

        self.num_batches = kwargs.get('num_batches', 10)
        self.rb = Buffer(capacity=self.num_batches * self.batch_size)

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
            try:
                prob = Categorical(logits=self.policy_model(obs_tensor.to(device)).view(-1))
            except Exception as exp:
                print(self.policy_model(obs_tensor.to(device)).view(-1))
                raise ValueError(exp)
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
                if self.critic_target == 'mc':
                    v = self.trajectory[i] + v * self.gamma
                elif self.critic_target == 'td':
                    v = self.trajectory[i]
                else:
                    raise ValueError(f'{self.critic_target} is not a valid critic target')
                self.rb.add((self.trajectory[i - 2],
                             self.trajectory[i - 1],
                             v,
                             self.trajectory[i + 1],
                             self.gamma ** (i // 3)))  # discounted to the starting point

        self.trajectory = []
        return

    def train(self):
        """
        train POLICY model !!!
        """

        if len(self.rb) < self.batch_size:
            return

        self.policy_model.train()

        opt_policy = torch.optim.RMSprop(self.policy_model.parameters(), lr=self.lr, eps=self.eps)
        opt_critic = torch.optim.RMSprop(self.critic_model.parameters(), lr=self.lr, eps=self.eps)

        logger.debug(f'====== Train PG critic={self.critic} (obs={len(self.rb)}) ======')

        t_v_loss, t_p_loss, counts = 0, 0, 0

        for (o, a, r, n_o, d) in self.rb.sample(self.batch_size):

            critic, v_loss = self.train_critic(o, a, r, n_o, d, opt_critic)

            opt_policy.zero_grad()
            p_loss = - (critic * (self.policy_model(o).softmax(dim=-1)[range(o.size(0)), a]).log()).mean()
            p_loss.backward()
            clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            opt_policy.step()

            t_v_loss += v_loss * len(a)
            t_p_loss += p_loss.item() * len(a)
            counts += len(a)

        t_v_loss /= counts
        t_p_loss /= counts
        logger.info(f'ValueLoss={t_v_loss:.3f}, PolicyLoss={t_p_loss:.3f}')

        self.policy_model.eval()

        return t_v_loss, t_p_loss

    def train_critic(self, o, a, r, n_o, d, opt_critic):
        """
        critic = mc/adv, the TD error=(TD-target - estimation) is also used for weighting PG
        critic = Q, TD error is only used for updating Q, Q is directly used for weighting PG
        """

        # get TD/MC target
        self.critic_model.eval()
        with torch.no_grad():
            if self.critic_target == 'td':
                v_next = self.critic_model(n_o.to(device)).cpu()
                if self.critic == 'q':
                    v_next = v_next[range(v_next.size(0)), a.to(device)]  # r + gamma*Q(s', a') SARSA target
                r += self.gamma * v_next.view(-1)  # r + gamma*V(s')

        # predict value
        self.critic_model.train()
        opt_critic.zero_grad()
        if self.critic == 'q':
            pred_v = self.critic_model(o.to(device))
            pred_v = pred_v[range(pred_v.size(0)), a.to(device)]
        else:
            pred_v = self.critic_model(o.to(device)).view(-1)

        v_loss = F.smooth_l1_loss(pred_v.to(device), r.to(device))
        v_loss.backward()
        clip_grad_norm_(self.critic_model.parameters(), self.max_grad_norm)
        opt_critic.step()

        # re-calculate the critic-estimation by the different train/eval behavior of the models
        # TD error
        self.critic_model.eval()
        with torch.no_grad():
            if self.critic == 'q':
                critic = self.critic_model(o.to(device))  # discounted Q function
                critic = critic[range(critic.size(0)), a.to(device)].view(-1) * d
            else:
                critic = (r - self.critic_model(o.to(device)).view(-1)) * d  # TD

            return critic, v_loss.item()

    def load_ckp(self, path, training=True):
        self.policy_model.load_state_dict(torch.load(f'{path}/policy.pth', map_location='cpu'))
        self.policy_model.eval()

        if training:
            try:
                self.critic_model.load_state_dict(torch.load(f'{path}/critic.pth', map_location='cpu'))
                self.policy_model.eval()
            except Exception as exp:
                raise ValueError(f'{exp}')

    def save_ckp(self, path):
        torch.save(self.policy_model.state_dict(), f'{path}/policy.pth')
        torch.save(self.critic_model.state_dict(), f'{path}/critic.pth')


if __name__ == '__main__':
    import gym

    frame_freq = 1
    history_len = 5

    env = gym.make('CartPole-v0')  # 'SpaceInvaders-v0'
    target = 'TD'
    # env = gym.make('Breakout-v0')
    agent = PGAgent(n_act=env.action_space.n,
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
