
import random
import torch
from torch import nn
from env.basic import GridWorldEnv
from agents import TabularGWAgent
from matplotlib import pyplot as plt
from tqdm import trange

"""
 off-line/MC methods does not works well if the max_len of episodes is small for windyGW
 (Reaching Terminal is very sparse and a random walk from S to T is very hard)
 If you wish to try off-line variates or MC on windyGW , please set max_len larger in Env.run() and use small 'update_freq'
 See Example 6.5 in Sutton's book for details
 """


class LinearGWAgent(TabularGWAgent):
    """
    online only
    """

    def __init__(self, n_action_space=4, sensor=False, q_back=False, **kwargs):
        super().__init__(n_action_space=n_action_space, **kwargs)

        self.name = 'linear_GW'
        self.sensor = sensor
        if sensor:
            self.input_c = 4
        else:
            self.input_c = self.grid_size[0] * self.grid_size[1]
        self.q_back = q_back
        self.eps_greedy = kwargs.get('eps_greedy', 0.1)
        self.policy_model = self.init_policy()

    def init_table(self):
        return {}

    def init_policy(self):
        """
        linear does not work
        be careful if you add too many layers (need clipping)
        """
        return nn.Sequential(
            nn.Linear(self.input_c, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_action_space)
        )

    def act(self, obs: dict, **kwargs):
        """
        const reward = -1
        """
        s = self.encode_s(obs)

        if self.training and random.random() < self.eps_greedy:
            a = random.choice(obs['la'])
        else:
            qs = self.predict(s)
            a = qs.argmax(dim=-1)

        if self.training:
            if self.q_back:
                self.trajectory += [-1, s, a] if self.trajectory else [s, a]
                return a
            if self.online:
                if self.pre_sa:
                    self.online_train(-1, s, a)
                self.pre_sa = (s, a)
        return a

    def predict(self, x):
        with torch.no_grad():
            return self.policy_model(x)

    def encode_s(self, obs: dict, **kwargs):
        """
        use one-hot
        linear kernel does not work well
        """
        vec = [0] * self.input_c
        vec[self.grid_size[1]*obs["pos"][0] + obs["pos"][1]] = 1

        return torch.FloatTensor(vec)

    def backup(self, payoff=0):
        """
        backup final immediate reward (payoff)
        constant reward = -1 for each step
        """
        if self.q_back and self.training:
            self.Q_back_propagation(payoff)
            return

        if self.online and self.training:
            self.online_train(payoff)

    def online_train(self, r, s=None, a=None):
        if not self.pre_sa:
            return
        self.policy_model.train()

        with torch.no_grad():
            p_s, p_a = self.pre_sa
            if s is None:
                y = r
            else:
                y = r + self.discount_factor * self.policy_model(s).max() if self.value_target == 'Q' \
                    else r + self.discount_factor * self.policy_model(s)[a]

        self.policy_model.zero_grad()
        loss = (self.policy_model(p_s)[p_a] - y) ** 2
        loss.backward()
        # gradient descent
        with torch.no_grad():
            for p in self.policy_model.parameters():
                p -= 2 * self.lr * p.grad
        self.policy_model.eval()

    def Q_back_propagation(self, final_payoff):
        if not self.trajectory:  # (s, a, r, s' a' ,r')
            return
        self.trajectory.append(final_payoff)
        ns, na = None, None
        for i in range(len(self.trajectory)-3, -1, -3):
            s, a, r = self.trajectory[i],  self.trajectory[i+1],  self.trajectory[i+2]
            self.pre_sa = (s, a)
            self.online_train(r, ns, na)
            ns, na = s, a

    def __str__(self):
        return f'{self.value_target}_{"online" if self.online else "offline"}'


def run_grid(World: GridWorldEnv, tabular=True, q_back=False, max_len=100, N_episodes=2000, lr=0.0005):
    update_freq = 1  # for offline only
    env = World

    methods = [('Q', True), ('sarsa', True)]
    # methods = [('Q', True), ('sarsa', True), ('Q', False), ('sarsa', False), ('mc', False)]
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    for (m, online) in methods:
        step, finished_episodes = 0, []
        tag = m + f'_{"online" if online else "offline"}'
        if tabular:
            agent = TabularGWAgent(grid_size=env.grid_size,
                                   training=True, value_target=m, online=online, lr=lr)
        else:
            agent = LinearGWAgent(grid_size=env.grid_size, training=True, q_back=q_back,
                                  value_target=m, online=True, lr=lr)
        print(agent)
        env.set_agent(agent)
        rec_payoff = []
        for episode in trange(N_episodes, desc=tag):
            agent.training = True
            obs, reward, done, info = env.reset()

            payoff = 0
            for _ in range(max_len):
                action = env.agent.act(obs)
                obs, reward, done, info = env.step(action)

                step += 1
                finished_episodes.append(episode)
                payoff += reward
                if done:
                    break

            agent.backup(reward)
            agent.reset()

            if episode % update_freq == 0:
                agent.offline_train()

            rec_payoff.append(payoff)

        # agent.save_model()
        """
        training results
        """
        ax[0].plot(range(len(rec_payoff)), rec_payoff, label=tag)
        ax[0].set_ylabel('Avg. score')
        ax[0].set_xlabel('Episodes')
        """
        reproduce results in Example 6.5
        """
        ax[1].plot(range(step), finished_episodes, label=tag)
        ax[1].set_ylabel('Episodes')
        ax[1].set_xlabel('Time steps')

        """
        plot policy
        """
        agent.training = False
        obs, reward, done, info = env.reset()
        path = [obs['pos'][:]]
        act = []
        di = ['^', 'v', '>', '<']
        payoff = 0
        for _ in range(100):
            action = env.agent.act(obs)
            act.append(di[action])
            obs, reward, done, info = env.step(action)
            path.append(obs['pos'][:])
            payoff += reward
            if done:
                break

        grids = env.get_raw_grid()
        for (i, j), a in zip(path, act):
            grids[j][i] = a

        grids = '\n'.join([env.__str__() + '_' +
                           agent.__str__() + f'_reward={payoff}'] + [' '.join(_) for _ in grids])
        print(grids)

    ax[0].set_title(env.__str__())
    ax[0].legend()
    ax[1].legend()
    fig.show()


if __name__ == '__main__':
    """
    why the algorithm not always finds the path during the testing phase?
    (randomness ?)
    """
    random.seed(0)
    from env import WindyGridWorldEnv, CliffWalkingEnv

    # tabular methods
    run_grid(WindyGridWorldEnv(), tabular=True, max_len=10000, N_episodes=100, lr=0.5)
    run_grid(CliffWalkingEnv(), tabular=True, max_len=100, N_episodes=2000, lr=0.5)

    # approximation methods
    # approx-sarsa is similar to tabular-sarsa, but approx-Q is slower than tabular-Q
    # unlike tabular-Q (guaranteed optimal), approx-Q is suboptimal OR optimal (randomness)
    # require high-representable-power model (e.g. hidden-dim=128)
    run_grid(WindyGridWorldEnv(), tabular=False, max_len=10000, N_episodes=100, lr=0.0005)
    run_grid(CliffWalkingEnv(), tabular=False, max_len=100, N_episodes=2000, lr=0.0005)

    # additional experiment: using Q-back_propagation, it's designed for finite horizon games
    # it's slow for games with too many episodes (windy GW..)
    run_grid(WindyGridWorldEnv(), tabular=False, q_back=True, max_len=10000, N_episodes=100, lr=0.0005)
    run_grid(CliffWalkingEnv(), tabular=False, q_back=True, max_len=100, N_episodes=2000, lr=0.0005)
