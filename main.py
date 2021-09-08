import gym
import time
from gym import spaces

import gym
import asyncio
from human import keyboard_act

# env = gym.make('CartPole-v0')


async def main():
    env = gym.make('MontezumaRevenge-v0')
    env.reset()  # reset for each new trial
    t = 0
    while True:
        env.render()
        action = await keyboard_act()
        if isinstance(action, int):
            obs, reward, done, info = env.step(action)
            print(reward)
        else:
            obs, reward, done, info = env.step(0)  # do nothing
        t += 1
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        time.sleep(0.05)


if __name__ == '__main__':
    asyncio.run(main())
