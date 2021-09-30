
import pickle
from matplotlib import pyplot as plt


# game = 'Breakout-v4'  # 'SpaceInvaders-ram-v4'  'CartPole-v0'  Breakout-v4
ckp_name = 'ts'

with open(f'checkpoints/{ckp_name}/performance.pickle', 'rb') as file:
    ckp = pickle.load(file)

with open(f'checkpoints/{ckp_name}/config.pickle', 'rb') as file:
    config = pickle.load(file)

loss, reward = ckp['loss_rec'], ckp['reward_rec']


fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].set_title(config['game'] + config['S'] + '-'+str(config['momentum']))
ax[0].plot(loss, label='loss')
ax[1].plot(reward, label='reward')
ax[0].set_xlabel('Num_updates')
ax[1].set_xlabel('Num_updates')
ax[0].legend()
ax[1].legend()
fig.show()



