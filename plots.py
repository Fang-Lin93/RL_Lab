
import json
from matplotlib import pyplot as plt

with open('results/CartPole-v1_loss.json', 'r') as file:
    loss = json.load(file)


plt.plot(loss, label='loss')
plt.legend()
plt.show()



