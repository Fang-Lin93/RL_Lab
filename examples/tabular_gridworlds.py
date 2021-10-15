
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


def run_grid(World: GridWorldEnv):
    N_episodes = 1000
    max_len = 100000
    update_freq = 1  # for offline only

    env = World

    methods = [('Q', True), ('sarsa', True)]
    # methods = [('Q', True), ('sarsa', True), ('Q', False), ('sarsa', False), ('mc', False)]
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    for (m, online) in methods:
        step, finished_episodes = 0, []
        tag = m + f'_{"online" if online else "offline"}'
        agent = TabularGWAgent(grid_size=env.grid_size,
                               training=True, value_target=m, online=online, lr=0.5)
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

        agent.save_model()
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
    import random
    random.seed(0)
    from env import WindyGridWorldEnv, CliffWalkingEnv

    run_grid(WindyGridWorldEnv())
    run_grid(CliffWalkingEnv())
