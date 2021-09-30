import gym
import time
import asyncio
from agents.human import HumanAtariAgent


# MontezumaRevenge-v0, Breakout-v4

async def main():
    env = gym.make('MontezumaRevenge-v4')
    obs = env.reset()  # reset for each new trial
    LA = list(range(env.action_space.n))
    state_dict = {
        'obs': obs,
        'la': LA,
        'reward': None,
        'done': False,
    }

    t = 0
    score = 0
    agent = HumanAtariAgent()
    while True:
        env.render()
        action = await agent.step(state_dict)
        obs, reward, done, info = env.act(action)
        t += 1
        state_dict = {
            'obs': obs,
            'la': LA,
            'reward': reward,
            'done': done,
        }
        score += reward
        if done:
            print(f"Episode finished after {t + 1} time steps total reward={score}")
            break
        time.sleep(0.05)

asyncio.run(main())
