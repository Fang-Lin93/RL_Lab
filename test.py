
import gym
import pickle
import random
from collections import deque

import torch
from loguru import logger
from agents import DQNAgent, PGAgent


# TODO: agent loading ckp methods
def eva(args, max_episode=1000):
    with open(f'checkpoints/{args.C}/performance.pickle', 'rb') as file:
        checkpoints = pickle.load(file)

    with open(f'checkpoints/{args.C}/config.pickle', 'rb') as file:
        config = pickle.load(file)

    logger.info(f'Episode={checkpoints["episode"]}')

    for k, v in config.items():
        logger.info(f'{k}={v}')

    env = gym.make(config['game'])
    obs = env.reset()
    history = deque([obs.tolist()+[0]], maxlen=config['history_len'])
    la = list(range(env.action_space.n))

    if args.A == 'dqn':
        agent = DQNAgent(**config)
    elif args.A == 'pg':
        agent = PGAgent(**config)
    else:
        raise ValueError(f'Agent {args.A} not found')

    agent.training = False
    agent.eps_greedy = -1

    agent.load_ckp(args.C)
    #
    # try:
    #     agent.policy_model.load_state_dict(torch.load(f'checkpoints/{args.C}/target.pth', map_location='cpu'))
    #     agent.policy_model.eval()
    # except Exception as exp:
    #     raise ValueError(f'{exp}')

    state_dict = {
        'obs': history,
        'la': la,
        'reward': None,
        'done': False,
    }
    t, score = 0, 0

    while True:
        env.render()
        action = agent.act(state_dict)

        obs, reward, done, info = env.step(action)
        t += 1
        history.append(obs.tolist()+[t/max_episode])

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
            agent.act(state_dict)
            logger.info(f"Episode finished after {t + 1} "
                        f"time steps with reward = {score} ")
            break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DQN test')

    parser.add_argument('--C', default='run', type=str)
    parser.add_argument('--A', default='dqn', type=str)
    args = parser.parse_args()
    eva(args)
