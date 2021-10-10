import random
from abc import ABC

import torch
from loguru import logger
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch import FloatTensor, LongTensor
from agents.basic import ReplayBuffer, BaseAgent
from agents.utils import process_image_obs, process_vec_obs
from torch.distributions import Categorical

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

EPS = 1e-5


class Buffer(ReplayBuffer):

    def sample(self, batch_size):
        random.shuffle(self.data)
        return (self.collate_fn(self.data[i:i + batch_size]) for i in range(0, len(self), batch_size))

    @staticmethod
    def collate_fn(batch):
        o, a, p, adv, r, d = [], [], [], [], [], []
        for item in batch:
            o.append(item[0])
            a.append(item[1])
            p.append(item[2])
            adv.append(item[3])
            r.append(item[4])
            d.append(item[5])
        return torch.stack(o), LongTensor(a),  torch.stack(p), FloatTensor(adv), FloatTensor(r), FloatTensor(d)


class PPOAgent(BaseAgent, ABC):
    """
    PPO agent
    use shared paras for policy and V(s)
    adv: advantage = r + V(s') - V(s)
    learn adv using either td or mc
    """

    critic_targets = ['mc', 'td']

    def __init__(self, critic_target='mc', **kwargs):
        super().__init__(name='ppo', **kwargs)

        self.model = self.init_model(value_head=True).to(device)
        self.c1 = kwargs.get('c1', 0.1)  # value loss
        self.c2 = kwargs.get('c2', 0.1)  # entropy bonus
        self.clip_eps = kwargs.get('clip_eps', 0.2)
        self.opt_epoch = kwargs.get('opt_epoch', 5)  # number epoch for updating
        self.kl_penalty = kwargs.get('kl_penalty', -1)

        logger.info(f'Num paras(policy)={self.model.num_paras()}')

        self.critic_target = critic_target

        self.num_batches = kwargs.get('num_batches', 10)
        self.rb = Buffer(capacity=self.num_batches * self.batch_size)

    def act(self, state: dict):

        obs_, la, r, finished = state['obs'], state['la'], state['reward'], state['done']

        obs_tensor = process_image_obs(obs_, self.history_len) if self.input_rgb else \
            process_vec_obs(obs_, self.history_len, self.input_c, self.disable_byte_norm, self.lstm)

        prob, value = self.forward(obs_tensor)
        action = Categorical(probs=prob.view(-1)).sample().item()

        # save trajectories
        if self.training:
            self.record(r, obs_tensor, (action, prob+EPS, value), finished)
        return action

    def forward(self, obs_tensor):
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(obs_tensor.to(device))
            return logits.softmax(dim=-1).view(-1), value.item()

    def backup(self):
        """
        If Q-MC, using MC-target, otherwise using TD target

        if LSTM -> a trajectory = a batch
        I can update the paras in the final stage (use total loss)
        """
        if self.trajectory:
            v, next_v = 0, None
            for i in range(len(self.trajectory) - 2, 0, -3):

                # self.trajectory[i] is the immediate reward
                action, prob, value = self.trajectory[i - 1]

                td_tar = self.trajectory[i] + self.gamma * next_v if next_v is not None else self.trajectory[i]
                next_v = value
                # Value target for critic
                if self.critic_target == 'mc':
                    v = self.trajectory[i] + v * self.gamma
                elif self.critic_target == 'td':
                    v = td_tar
                else:
                    raise ValueError(f'{self.critic_target} is not a valid critic target')

                # advantage = td_tar - value
                self.rb.add((self.trajectory[i - 2],
                             action, prob, td_tar - value, v,
                             self.gamma ** (i // 3)))  # discounted to the starting point

        self.trajectory = []
        return

    def train(self):
        """
        train POLICY model !!!
        """

        if len(self.rb) < self.batch_size:
            return

        self.model.train()

        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, eps=self.eps)

        logger.debug(f'====== Train PPO (obs={len(self.rb)}) ======')

        t_loss, t_ppo_target, t_v_loss, t_entropy, counts = 0, 0, 0, 0, 0

        for epoch in range(self.opt_epoch):
            for (o, a, p, adv, r, d) in self.rb.sample(self.batch_size):
                optimizer.zero_grad()
                logits, val = self.model(o)
                prob = logits.softmax(dim=-1)
                ratio = prob[range(logits.size(0)), a] / p[range(p.size(0)), a]

                if self.clip_eps > 0:
                    clip_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    ppo_target = (torch.min(ratio * adv, clip_ratio * adv) * d).mean()
                else:
                    ppo_target = (ratio * adv * d).mean()

                val_loss = 0.5 * (val - r).pow(2).mean()
                entropy = Categorical(logits=logits).entropy().mean()

                loss = - ppo_target + self.c1 * val_loss - self.c2 * entropy

                if self.kl_penalty > 0:  # KL-penalty
                    loss += self.kl_penalty * (prob * (prob/p).log()).sum()

                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()

                t_loss += loss.item() * len(a)
                t_ppo_target += ppo_target.item() * len(a)
                t_v_loss += val_loss.item() * len(a)
                t_entropy += entropy.item() * len(a)
                counts += len(a)

            logger.info(f'Epoch[{epoch}/{self.opt_epoch}]: (avg. stat.)\n'
                        f'Total Loss={t_loss / counts:.3f}\t'
                        f'PPO Target={t_ppo_target / counts:.3f}\t'
                        f'Value Loss={t_v_loss / counts:.3f}\t'
                        f'Entropy={t_entropy / counts:.3f}')

        self.model.eval()
        return t_loss / counts, t_ppo_target / counts, t_v_loss / counts, t_entropy / counts  # final statistics

    def load_ckp(self, path):
        self.model.load_state_dict(torch.load(f'{path}/model.pth', map_location='cpu'))

    def save_ckp(self, path):
        torch.save(self.model.state_dict(), f'{path}/model.pth')


if __name__ == '__main__':
    import gym

    frame_freq = 1
    history_len = 5

    env = gym.make('CartPole-v0')  # 'SpaceInvaders-v0'
    target = 'TD'
    # env = gym.make('Breakout-v0')
    agent = PPOAgent(n_act=env.action_space.n,
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
