
import pickle
from matplotlib import pyplot as plt


# game = 'Breakout-v4'  # 'SpaceInvaders-ram-v4'  'CartPole-v0'  Breakout-v4
ckp_name = 'cppg'

with open(f'checkpoints/{ckp_name}/performance.pickle', 'rb') as file:
    ckp = pickle.load(file)

with open(f'checkpoints/{ckp_name}/config.pickle', 'rb') as file:
    config = pickle.load(file)

# loss, reward = ckp['loss_rec'], ckp['reward_rec']

fig, ax = plt.subplots(len(ckp)-1, 1, figsize=(10, 10 + len(ckp)))
ax[0].set_title(config['game'] + config['S'])
del ckp['episode']
for i, (k, v) in enumerate(ckp.items()):

    ax[i].plot(v, label=k)
    ax[i].set_xlabel('Num_updates')
    ax[i].legend()
    # ax[1].plot(reward, label='reward')
    # ax[0].set_xlabel('Num_updates')
    # ax[1].set_xlabel('Num_updates')
    # ax[0].legend()
    # ax[1].legend()
fig.show()



