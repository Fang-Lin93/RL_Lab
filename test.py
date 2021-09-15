
import gym
import json
from loguru import logger
from agents.dqn import DQNAgent


def eva(game: str, max_episode=1000, model_file: str = 'v0'):
    env = gym.make(game)
    obs = env.reset()
    la = list(range(env.action_space.n))

    with open(f'results/{game}.json', 'r') as file:
        model_config = json.load(file)

    for k, v in model_config.items():
        logger.info(f'{k}={v}')

    agent = DQNAgent(**model_config)
    agent.training = False
    agent.eps_greedy = -1

    try:
        agent.policy_model.load_model(f'{game}_{model_file}')
    except Exception as exp:
        raise ValueError(f'{exp}, model dqn_{game}_{model_file}.pth not find')

    history = [obs]
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
        if len(history) > model_config['history_len']:
            history.pop(0)

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

    parser.add_argument('--game', default='Breakout-ram-v4', type=str)
    # SpaceInvaders-v0 Breakout-v4 CartPole-v1  Breakout-ramNoFrameskip-v4 Breakout-ram-v4
    args = parser.parse_args()
    eva(game=args.game)
