
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
    t, score, frame = 0, 0, 0
    action = None

    while True:
        env.render()
        if frame == 0:
            action = agent.step(state_dict)
            frame = model_config['frame_freq'] - 1

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
        frame -= 1
        if done or t > max_episode:
            state_dict['done'] = True
            agent.step(state_dict)
            logger.info(f"Episode finished after {t + 1} "
                        f"time steps with reward = {score} ")
            break


if __name__ == '__main__':

    # SpaceInvaders-v0  Breakout-v0 CartPole-v0
    # eva(game='SpaceInvaders-v0')
    eva(game='Breakout-ram-v0')
