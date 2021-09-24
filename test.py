
import gym
import pickle
import random
from collections import deque

import torch
from loguru import logger
from agents.dqn import DQNAgent


def eva(ckp: str, max_episode=1000, model_file: str = 'v0'):
    with open(f'checkpoints/{ckp}/ckp.pickle', 'rb') as file:
        checkpoints = pickle.load(file)

    with open(f'checkpoints/{ckp}/config.pickle', 'rb') as file:
        config = pickle.load(file)

    logger.info(f'Episode={checkpoints["episode"]}')

    for k, v in config.items():
        logger.info(f'{k}={v}')

    env = gym.make(config['game'])
    obs = env.reset()
    history = deque([obs], maxlen=config['history_len'])
    la = list(range(env.action_space.n))

    agent = DQNAgent(**config)
    agent.training = False
    agent.eps_greedy = -1

    try:
        agent.policy_model.load_state_dict(torch.load(f'checkpoints/{ckp}/target.pth', map_location='cpu'))
        agent.policy_model.eval()
    except Exception as exp:
        raise ValueError(f'{exp}')

    state_dict = {
        'obs': history,
        'la': la,
        'reward': None,
        'done': False,
    }
    t, score = 0, 0

    while True:
        env.render()
        action = agent.step(state_dict)

        obs, reward, done, info = env.step(action)

        history.append(obs)

        state_dict = {
            'obs': history,
            'la': la,
            'reward': reward,
            'done': False
        }

        logger.debug(f'Action={action}, R_t+1={reward}')
        score += reward
        t += 1
        if done or t > max_episode:
            state_dict['done'] = True
            agent.step(state_dict)
            logger.info(f"Episode finished after {t + 1} "
                        f"time steps with reward = {score} ")
            break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DQN test')

    parser.add_argument('--ckp', default='run', type=str)
    args = parser.parse_args()
    eva(ckp=args.ckp)
