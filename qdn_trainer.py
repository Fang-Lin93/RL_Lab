
import gym
import sys
from matplotlib import pyplot as plt
from loguru import logger
from agents.dqn import DQNAgent
import argparse
parser = argparse.ArgumentParser(description='DQN')

logger.remove()
logger.add(sys.stderr, level='INFO')

#
# N_epochs = 10
# buffer_size = 10000
# eps_greedy = 0.1
# gamma = 0.9
# batch_size = 64
# max_grad_norm = 40
# render = 'human'  # 'rgb_array'  # 'human'
# game = 'SpaceInvaders-v0'
# target = 'TD'

parser.add_argument('--N_epochs', default=10, type=int, help='N_epochs (default: 10)')
parser.add_argument('--buffer_size', default=10000, type=int, help='buffer_size (default: 10000)')
parser.add_argument('--eps_greedy', default=0.1, type=float, help='eps_greedy (default: 0.1)')
parser.add_argument('--gamma', default=0.9, type=float, help='decay factor (default: 0.9)')
parser.add_argument('--batch_size', default=256, type=int, help='batch_size (default: 256)')
parser.add_argument('--max_grad_norm', default=40, type=float, help='max_grad_norm for clipping grads (default: 40)')
parser.add_argument('--render', default='rgb_array', type=str, help='where to show? (human/rgb_array) (default: "human")')
parser.add_argument('--game', default='SpaceInvaders-v0', type=str, help='game env name')
parser.add_argument('--target', default='TD', type=str, help='target = TD/MC (default: TD)')

args = parser.parse_args()


def main():
    env = gym.make(args.game)
    agent = DQNAgent(n_act=env.action_space.n,
                     training=True,
                     eps_greedy=args.eps_greedy,
                     gamma=args.gamma,
                     batch_size=args.batch_size,
                     buffer_size=args.buffer_size,
                     max_grad_norm=args.max_grad_norm,
                     target_type=args.target)
    score_recorder = []
    for epoch in range(args.N_epochs):
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
            env.render(args.render)
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

        agent.policy_model.save_model(f'{args.game}_v0')

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
