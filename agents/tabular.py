
import json
import os
import random
from loguru import logger
from collections import deque


class QTableAgent(object):
    """
    discount factor = 1
    use on-line -> train while running
    use off-line -> use replay buffer to train
    """
    def __init__(self, n_action_space, **kwargs):
        self.name = 'Q_table'
        self.n_action_space = n_action_space

        self.eps_greedy = kwargs.get('eps_greedy', 0.1)
        self.training = kwargs.get('training', False)
        self.value_target = kwargs.get('value_target', 'mc')  # either 'mc', 'sarsa'(td), 'Q'
        self.buffer_size = kwargs.get('buffer_size', 10000)
        self.online = kwargs.get('online', False)  # on-line/off-line training
        self.lr = kwargs.get('lr', 0.1)
        self.discount_factor = kwargs.get('discount_factor', 1)

        if self.online:
            if self.value_target == 'mc':
                logger.warning('MC-target is not acceptable for on-line learning. The learning mode is set to off-line')
                self.online = False

        # on-line training
        self.pre_sa = None

        # off-line training
        self.rb = deque([], maxlen=self.buffer_size)  # replay buffer
        self.trajectory = []

        # Q tables
        self.Q_table = {}  # policy Q
        self.target_Q = {}  # only for Q-learning (delay updates)
        self.init_table()
        if self.value_target == 'Q':
            self.sync_Q()

    def reset(self):
        self.pre_sa = ()
        self.trajectory = []

    def sync_Q(self):
        self.target_Q = {k: v[:] for k, v in self.Q_table.items()}

    def init_table(self):
        """
        initial a Q-table with random/pre-determined values.
        The key of Q table is a string with form f'{state}_{action}' action in range(self.n_action_space)

        str(state) should be consist with the QTableAgent.encode_s method

        Example:
            In Blackjack,

         for _d_c in range(1, 11):
            for _s in range(12, 22):
                self.Q_table[f'{_d_c}-{_s}_0'] = [random.uniform(-1, 1) for _ in range(self.n_action_space)]
                self.Q_table[f'{_d_c}-{_s}_1'] = [random.uniform(-1, 1) for _ in range(self.n_action_space)]

        """
        raise NotImplementedError()

    def act(self, obs: dict, **kwargs):
        """
        the main function for taking actions

        Example:

        if not obs['finished'][player_id]:
            s = self.encode_s(obs, player_id=player_id)

            if self.training and random.random() < self.eps_greedy:
                a = random.choice(obs['la'])
            else:  # choose the largest one
                qs = self.Q_table[s] if not self.require_delayed_update() else self.target_Q[s]
                a = qs.index(max(qs))

            if self.training:
                if self.online:
                    if self.pre_sa:
                        self.online_train(0, s, a)
                    self.pre_sa = (s, a)
                else:
                    self.trajectory += [s, a]

            return a
        return
        """
        raise NotImplementedError()

    @staticmethod
    def encode_s(obs: dict, **kwargs) -> str:
        """
        encode states to a string representation
        """
        raise NotImplementedError()

    def offline_train(self):
        if self.online:
            return
        if self.rb:
            """
            batch average
            """
            batch_y = {}
            for (s, a, r, ns) in self.rb:
                if (s, a) not in batch_y:
                    batch_y[(s, a)] = [0, 1]  # (value, cnt)
                else:
                    batch_y[(s, a)][1] += 1
                if self.value_target == 'Q' and ns is not None:
                    batch_y[(s, a)][0] += r + self.discount_factor * max(self.target_Q[ns])
                else:
                    batch_y[(s, a)][0] += r

            for (s, a) in batch_y.keys():
                self.Q_table[s][a] = self.Q_table[s][a] + \
                                     self.lr * (batch_y[(s, a)][0]/batch_y[(s, a)][1] - self.Q_table[s][a])

            if self.value_target == 'Q':
                self.sync_Q()
            else:
                self.rb = deque([], maxlen=self.buffer_size)

    def online_train(self, r=0, s=None, a=None):
        if not self.pre_sa:
            return
        p_s, p_a = self.pre_sa
        if s is None:
            y = r
        else:
            y = r + self.discount_factor * max(self.Q_table[s]) if self.value_target == 'Q' \
                else r + self.discount_factor * self.Q_table[s][a]

        self.Q_table[p_s][p_a] = self.Q_table[p_s][p_a] + self.lr * (y - self.Q_table[p_s][p_a])

    def backup(self, payoff):
        if self.online and self.training:
            """
            backup the final reward and train if online
            """
            self.online_train(payoff)
        else:
            """
            save the current trajectory to the replay buffer
            
            Example:
            
            for i in range(0, len(self.trajectory), 2):  # (s, a, s', a')
                s, a = (self.trajectory[i], self.trajectory[i + 1])

                if self.value_target == 'Q':  # (s, a, s'/G )
                    self.rb.append((s, a, self.trajectory[i + 2] if i + 2 < len(self.trajectory) else payoff))

                elif self.value_target == 'sarsa':  # (s, a, target_value )
                    if i + 3 < len(self.trajectory):
                        tar = self.discount_factor * self.Q_table[self.trajectory[i + 2]][self.trajectory[i + 3]]
                    else:
                        tar = payoff
                    self.rb.append((s, a, tar))

                elif self.value_target == 'mc':
                    self.rb.append((s, a, payoff))

                else:
                    raise ValueError(f'value target should be either td or mc, not {self.value_target}')

            """
            raise NotImplementedError()

    def save_model(self):
        if not os.path.exists('results'):
            os.mkdir('results')
        with open(f'results/{self.name}_{self.value_target}_{"online"if self.online else "offline"}.json', 'w') as file:
            json.dump(self.target_Q, file) if self.require_delayed_update() else json.dump(self.Q_table, file)

    def load_model(self):
        with open(f'results/{self.name}_{self.value_target}_{"online"if self.online else "offline"}.json', 'r') as file:
            self.Q_table = json.load(file)
            if self.require_delayed_update():
                self.sync_Q()

    def require_delayed_update(self):
        return self.value_target == 'Q' and not self.online


class TabularGWAgent(QTableAgent):
    """
    discount factor = 1
    use on-line -> train while running
    use off-line -> use replay buffer to train
    """

    def __init__(self, grid_size, n_action_space=4, **kwargs):
        self.grid_size = grid_size

        super().__init__(n_action_space, **kwargs)
        self.name = 'grid_world agent'

    def init_table(self):
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                self.Q_table[f'{x}-{y}'] = [0 for _ in range(self.n_action_space)]

    def act(self, obs: dict, **kwargs):
        """
        const reward = -1
        """
        s = self.encode_s(obs)

        if self.training and random.random() < self.eps_greedy:
            a = random.choice(obs['la'])
        else:
            qs = self.Q_table[s] if not self.require_delayed_update() else self.target_Q[s]
            a = qs.index(max(qs))

        if self.training:
            if self.online:
                if self.pre_sa:
                    self.online_train(-1, s, a)
                self.pre_sa = (s, a)
            else:
                self.trajectory += [s, a]

        return a

    @staticmethod
    def encode_s(obs: dict, **kwargs):
        return f'{obs["pos"][0]}-{obs["pos"][1]}'

    def backup(self, payoff=0):
        """
        backup final immediate reward (payoff)
        constant reward = -1 for each step
        """
        if self.online and self.training:
            self.online_train(payoff)
        else:
            for i in range(len(self.trajectory) - 2, -1, -2):  # (s, a, s', a')
                s, a = (self.trajectory[i], self.trajectory[i + 1])

                if self.value_target == 'Q':  # (s, a, s'/G )
                    if i + 2 < len(self.trajectory):
                        self.rb.append((s, a, -1, self.trajectory[i + 2]))
                    else:
                        self.rb.append((s, a, payoff, None))  # terminal

                elif self.value_target == 'sarsa':  # (s, a, target_value )
                    if i + 3 < len(self.trajectory):
                        tar = self.discount_factor * self.Q_table[self.trajectory[i + 2]][self.trajectory[i + 3]] - 1
                    else:
                        tar = payoff
                    self.rb.append((s, a, tar, None))

                elif self.value_target == 'mc':
                    self.rb.append((s, a, payoff, None))
                    payoff = self.discount_factor * payoff - 1

                else:
                    raise ValueError(f'value target should be either td or mc, not {self.value_target}')

    def __str__(self):
        return f'{self.value_target}_{"online" if self.online else "offline"}'

