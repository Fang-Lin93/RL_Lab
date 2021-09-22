
import gym
import pickle
import random
from collections import deque
from loguru import logger
from agents.dqn import DQNAgent


def eva(ckp: str, max_episode=1000, model_file: str = 'v0'):
    with open(f'checkpoints/{ckp}_ckp.pickle', 'rb') as file:
        checkpoints = pickle.load(file)

    config = checkpoints['config']


    for k, v in config.items():
        logger.info(f'{k}={v}')

    env = gym.make(config['game'])
    obs = env.reset()
    history = deque([obs], maxlen=config['history_len'])
    la = list(range(env.action_space.n))

    # for _ in range(random.randint(1, config['no_op_max'])):
    #     obs, _, _, _ = env.step(1)  # force game start !
    #     history.append(obs)

    agent = DQNAgent(**config)
    agent.training = False
    agent.eps_greedy = -1

    try:
        agent.policy_model.load_state_dict(checkpoints['model_dict']['target'])
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

    parser.add_argument('--game', default='Breakout-v4', type=str)
    parser.add_argument('--ckp', default='run', type=str)
    # SpaceInvaders-ram-v4 Breakout-v4 CartPole-v1  Breakout-ramNoFrameskip-v4 Breakout-ram-v4
    args = parser.parse_args()
    eva(ckp=args.ckp)
