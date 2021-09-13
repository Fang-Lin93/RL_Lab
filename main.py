import time
from gym import spaces
import gym
import asyncio
# from human import keyboard_act
from agents.human import HumanAtariAgent
from agents.rand import RandomAgent

#  MontezumaRevenge-v0  SpaceInvaders-v0


async def async_main():
    env = gym.make('MontezumaRevenge-v0')
    obs = env.reset()  # reset for each new trial
    state = {
        'obs': [obs],
        'la': list(range(env.action_space.n)),
        'reward': None
    }
    t = 0
    agent = HumanAtariAgent()
    while True:
        env.render()
        action = await agent.step(state)
        obs, reward, done, info = env.step(action)
        state = {
            'obs': [obs],
            'la': list(range(env.action_space.n)),
            'reward': reward,
        }
        t += 1
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        time.sleep(0.05)


def main(agent, max_episode=1000):
    env = gym.make('SpaceInvaders-v0')
    obs = env.reset()  # reset for each new trial
    la = list(range(env.action_space.n))
    state = {
        'obs': [obs],
        'la': la,
        'reward': None
    }
    t = 0
    while True:
        env.render()
        action = agent.step(state)
        obs, reward, done, info = env.step(action)
        state = {
            'obs': [obs],
            'la': la,
            'reward': reward,
        }
        print(f'Action={action}, R_t+1={reward}')
        t += 1
        if done or t > max_episode:
            agent.step(state)  # backup the final state
            print("Episode finished after {} timesteps".format(t + 1))
            break


if __name__ == '__main__':

    # import asyncio
    # asyncio.run(async_main())
    # main(agent=RandomAgent())

    from agents.dqn import DQNAgent

    main(agent=DQNAgent(n_act=6, eps_greedy=0.2))








