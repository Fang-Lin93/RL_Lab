
import gym
import sys
from matplotlib import pyplot as plt
from loguru import logger
from agents.dqn import DQNAgent
logger.remove()
logger.add(sys.stderr, level='INFO')


N_epochs = 10
buffer_size = 10000
eps_greedy = 0.1
gamma = 0.9
batch_size = 64
max_grad_norm = 40
render_mode = 'human'  # 'rgb_array'  # 'human'
game = 'SpaceInvaders-v0'
target_type = 'TD'


def main():
    env = gym.make(game)
    agent = DQNAgent(n_act=env.action_space.n,
                     training=True,
                     eps_greedy=eps_greedy,
                     gamma=gamma,
                     batch_size=batch_size,
                     buffer_size=buffer_size,
                     max_grad_norm=max_grad_norm,
                     target_type=target_type)
    score_recorder = []
    for epoch in range(N_epochs):
        logger.info(f'Epoch={epoch}')
        obs = env.reset()
        agent.reset()
        state_dict = {
            'obs': [obs],
            'la': list(range(env.action_space.n)),
            'reward': None,
            'done': False,
        }
        t, score, frame = 0, 0, 4
        action = None

        while True:
            env.render(render_mode)
            if frame == 4:
                action = agent.step(state_dict)
                frame = 0

            obs, reward, done, info = env.step(action)
            state_dict = {
                'obs': [obs],
                'la': list(range(env.action_space.n)),
                'reward': reward,
                'done': False
            }

            logger.debug(f'Action={action}, R_t+1={reward}')
            score += reward
            t += 1
            frame += 1
            if done or t > 1000:
                state_dict['done'] = True
                agent.step(state_dict)
                logger.info(f"Episode finished after {t + 1} time steps with reward = {score}")
                break

        score_recorder.append(score)

        agent.process_trajectory()

        agent.train_loop()
        agent.sync_model()

        agent.policy_model.save_model(f'{game}_v0')

    plt.plot(score_recorder, label='score')
    plt.savefig('res.png')
    plt.show()
    return


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser(description='DQN')
    #
    # # Save & Game config
    # parser.add_argument('--train_device', default='cuda:0', type=str,
    #                     help='Device for training (default: cuda:0)')

    main()
