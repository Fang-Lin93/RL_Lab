
import pickle
from matplotlib import pyplot as plt


# game = 'Breakout-v4'  # 'SpaceInvaders-ram-v4'  'CartPole-v0'  Breakout-v4
def plot_ckp(ckp_name):
    with open(f'checkpoints/{ckp_name}/performance.pickle', 'rb') as file:
        ckp = pickle.load(file)

    with open(f'checkpoints/{ckp_name}/config.pickle', 'rb') as file:
        config = pickle.load(file)

    x = [round(_ / 60, 2) for _ in ckp['time']]
    del ckp['episode'], ckp['time'], ckp['start_time']

    fig, ax = plt.subplots(len(ckp), 1, figsize=(10, 8 + len(ckp)))
    ax[0].set_title(f'{config["game"]}_{config["S"]}')
    for i, (k, v) in enumerate(ckp.items()):
        ax[i].plot(x, v, label=k)
        ax[i].legend()
    ax[-1].set_xlabel('Time used (in min)')
    fig.show()


if __name__ == '__main__':
    plot_ckp('ppo')





