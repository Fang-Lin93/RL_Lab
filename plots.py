
import pickle
from matplotlib import pyplot as plt


game = 'Breakout-ram-v4'  # 'SpaceInvaders-ram-v4'  'CartPole-v1'

with open(f'results/{game}_loss.pickle', 'rb') as file:
    loss = pickle.load(file)

with open(f'results/{game}_reward.pickle', 'rb') as file:
    reward = pickle.load(file)


fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(loss, label='loss')
ax[1].plot(reward, label='reward')
ax[0].legend()
ax[1].legend()
fig.show()



