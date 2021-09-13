import keyboard
import time

from consts import KEYBOARD_MAP


# async def keyboard_act():
#     for key, v in KEYBOARD_MAP.items():
#         if keyboard.is_pressed(key):
#             print(f'You Pressed {key}')
#             return v


class HumanAtariAgent(object):
    """
    for Atari
    """
    def __init__(self):
        self.name = 'human'

    async def step(self, state: dict):
        la = state['legal_actions']
        act = self.keyboard_act()
        if act in la:
            return act
        return 0

    @staticmethod
    def keyboard_act():
        for key, v in KEYBOARD_MAP.items():
            if keyboard.is_pressed(key):
                print(f'You Pressed {key}')
                return v
        return 0

#
# if __name__ == '__main__':
#
#     import gym
#     import asyncio
#
#     # env = gym.make('CartPole-v0')
#
#     async def main():
#         env = gym.make('MontezumaRevenge-v0')
#         env.reset()  # reset for each new trial
#         t = 0
#         agent = HumanAtariAgent()
#         while True:
#             env.render()
#             action = await agent.step()
#             obs, reward, done, info = env.step(action)
#             t += 1
#             if done:
#                 print("Episode finished after {} timesteps".format(t + 1))
#                 break
#             time.sleep(0.05)
#
#     asyncio.run(main())




