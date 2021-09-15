import gym
import sys
import time
import json
import os
from matplotlib import pyplot as plt
from loguru import logger
from agents.dqn import DQNAgent
import argparse

parser = argparse.ArgumentParser(description='DQN')

"""
using RNN to encode the hidden state requires full trajectories.
So we cannot simply sample (s, a, r, s') and make TD updates
here (o, h, c, a, r, no, nh, nc) is given and 'h, c, nh, nc' are already calculated by the previous model
Once the model is synchronized, the replay buffer should be cleared !

A normal DQN requires (s, a, r, s'), which cannot use RNN as state represents, or you can use 'history' as s
"""

parser.add_argument('--N_episodes', default=1000000, type=int, help='N_episodes')
parser.add_argument('--max_len', default=10000, type=int, help='max_len of episodes')
parser.add_argument('--buffer_size', default=1000000, type=int, help='buffer_size of trajectory')
parser.add_argument('--eps_greedy', default=0.1, type=float, help='eps_greedy (default: 0.1)')
parser.add_argument('--hidden_size', default=128, type=int, help='hidden_size')
parser.add_argument('--anneal_greedy', default=0.99, type=float, help='eps_greedy')
parser.add_argument('--gamma', default=0.99, type=float, help='decay factor')

parser.add_argument('--batch_size', default=256, type=int, help='batch_size')

parser.add_argument('--max_grad_norm', default=10, type=float, help='max_grad_norm for clipping grads (default: 40)')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate of RMSProp (default: 0.0001)')
parser.add_argument('--eps', default=1e-5, type=float, help='eps of RMSProp  (default: 1e-5)')

parser.add_argument('--target', default='TD', type=str, help='target = TD/MC, MC only in short episodes (default: TD)')
parser.add_argument('--render', default='rgb_array', type=str, help='where to show? (human/rgb_array)')
parser.add_argument('--train_freq', default=10, type=int, help='train every ? episode (default: 1)')
parser.add_argument('--frame_freq', default=3, type=int, help='act every ? frame')

parser.add_argument('--history_len', default=50, type=int, help='length of the history used, left zeros')

parser.add_argument('--game', default='SpaceInvaders-ram-v0', type=str, help='game env name')

#  Breakout-ram-v0  SpaceInvaders-ram-v0


args = parser.parse_args()


def main():

    env = gym.make(args.game)
    model_config = args.__dict__
    model_config.update({
        'n_act': env.action_space.n,
        'input_c': env.observation_space.shape[0],
        'input_rgb': 'ram' not in args.game,
    })
    for k, v in model_config.items():
        logger.info(f'{k}={v}')

    if not os.path.exists('results/'):
        os.mkdir('results/')

    with open(f'results/{args.game}.json', 'w') as handle:
        json.dump(model_config, handle)

    agent = DQNAgent(n_act=env.action_space.n,
                     input_c=env.observation_space.shape[0],
                     input_rgb='ram' not in args.game,
                     training=True,
                     eps_greedy=args.eps_greedy,
                     gamma=args.gamma,
                     batch_size=args.batch_size,
                     buffer_size=args.buffer_size,
                     max_grad_norm=args.max_grad_norm,
                     target_type=args.target,
                     lr=args.lr,
                     eps=args.eps,
                     hidden_size=args.hidden_size,
                     history_len=args.history_len)
    score_recorder = []
    s_time = time.time()

    for episode in range(args.N_episodes):
        logger.info(f'Epoch={episode}')
        obs = env.reset()
        agent.reset()

        agent.eps_greedy = max(args.eps_greedy, args.anneal_greedy ** episode)  # anneal epsilon greedy

        history = [obs]
        state_dict = {
            'obs': history,
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
                frame = args.frame_freq - 1

            obs, reward, done, info = env.step(action)

            history.append(obs)
            if len(history) > args.history_len:
                history.pop(0)

            state_dict = {
                'obs': history,
                'la': list(range(env.action_space.n)),
                'reward': reward,
                'done': False
            }

            logger.debug(f'Action={action}, R_t+1={reward}')
            score += reward
            t += 1
            frame -= 1
            if done or t > args.max_len:
                state_dict['done'] = True
                agent.step(state_dict)
                logger.info(f"Episode finished after {t + 1} "
                            f"time steps with reward = {score} "
                            f"remaining={(time.time() - s_time) / (episode + 1) * (args.N_episodes - episode - 1):.3f}s")
                break

        score_recorder.append(score)
        if args.target == 'TD':
            agent.process_trajectory(final_payoff=reward)
        if args.target == 'MC':
            agent.process_trajectory(final_payoff=score)

        if episode % args.train_freq == 0:
            # it may blow up if train too frequently
            agent.train_loop()
            agent.sync_model()

        agent.target_model.save_model(f'{args.game}_v0')

    plt.plot(score_recorder, label='score')
    plt.savefig('results/QDN_res.png')
    plt.show()
    return


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stderr, level='INFO')
    logger.add(f'DQN_{args.game}.log', level='INFO')

    main()
