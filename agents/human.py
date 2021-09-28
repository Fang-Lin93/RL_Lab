import keyboard
import time

# W, S, A, D, F as directions, I,K as angle_fires

KEYBOARD_MAP = {
     'n': 0,
     'f': 1,
     'w': 2,
     'd': 3,
     'a': 4,
     's': 5,
     'e': 6,
     'q': 7,
     'c': 8,
     'z': 9,
     'i': 10,
     'l': 11,
     'j': 12,
     'k': 13,
     'o': 14,
     'u': 15,
     ',': 16,
     'm': 17}


class HumanAtariAgent(object):
    """
    for Atari
    """

    def __init__(self):
        self.name = 'human'

    async def step(self, la: list = None):
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


if __name__ == '__main__':
    import gym
    import asyncio
    import argparse
    parser = argparse.ArgumentParser(description='Human')
    parser.add_argument('--g', default='MontezumaRevenge-v0', type=str)

    args = parser.parse_args()

    async def main():
        env = gym.make(args.g)
        env.reset()
        la = list(range(env.action_space.n))
        t = 0
        agent = HumanAtariAgent()
        while True:
            env.render()
            action = await agent.step(la)
            obs, reward, done, info = env.step(action)
            t += 1
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
            time.sleep(0.05)
    asyncio.run(main())

