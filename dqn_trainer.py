
import gym
import sys
import time
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

parser.add_argument('--N_episodes', default=1000, type=int, help='N_episodes (default: 10)')
parser.add_argument('--buffer_size', default=10000, type=int, help='buffer_size (default: 10000)')
parser.add_argument('--eps_greedy', default=0.1, type=float, help='eps_greedy (default: 0.1)')
parser.add_argument('--gamma', default=0.9, type=float, help='decay factor (default: 0.9)')
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--max_grad_norm', default=40, type=float, help='max_grad_norm for clipping grads (default: 40)')
parser.add_argument('--render', default='rgb_array', type=str, help='where to show? (human/rgb_array)')
parser.add_argument('--game', default='SpaceInvaders-v0', type=str, help='game env name')
parser.add_argument('--target', default='TD', type=str, help='target = TD/MC (default: TD)')
parser.add_argument('--train_freq', default=1, type=int, help='train every ? episode (default: 1)')
parser.add_argument('--frame_freq', default=2, type=int, help='act every ? frame (default: 2)')

args = parser.parse_args()


def main():

    for k, v in args.__dict__.items():
        logger.info(f'{k}={v}')

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
    s_time = time.time()

    for episode in range(args.N_episodes):
        logger.info(f'Epoch={episode}')
        obs = env.reset()
        agent.reset()
        state_dict = {
            'obs': [obs],
            'la': list(range(env.action_space.n)),
            'reward': None,
            'done': False,
        }
        t, score, frame = 0, 0, 0
        action = None

        while True:
            env.render(args.render)
            if frame == 0:
                action = agent.step(state_dict)
                frame = args.frame_freq

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
            frame -= 1
            if done or t > 1000:
                state_dict['done'] = True
                agent.step(state_dict)
                logger.info(f"Episode finished after {t + 1} "
                            f"time steps with reward = {score} "
                            f"remaining={(time.time()-s_time)/(episode+1)*(args.N_episodes-episode-1):.3f}s")
                break

        score_recorder.append(score)

        agent.process_trajectory()

        if episode % args.train_freq == 0:
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
