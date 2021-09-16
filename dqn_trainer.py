import gym
import sys
import time
import pickle
import os
from collections import deque
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

v0: 0.25 chance take previous action
v4: all action counted
Deterministic: skip 4 frames, otherwise randomly in (2, 5)
NoFrameskip-v4: no frame skip and no action repeat stochasticity
"""

# buffer
parser.add_argument('--N_episodes', default=100000, type=int, help='N_episodes')
parser.add_argument('--max_len', default=100000, type=int, help='max_len of episodes')
parser.add_argument('--buffer_size', default=10000, type=int, help='buffer_size of trajectory')
parser.add_argument('--eps_greedy', default=0.05, type=float, help='eps_greedy (default: 0.1)')
parser.add_argument('--explore_step', default=1000, type=int, help='anneal greedy')

# model  lstm can easily blow up
parser.add_argument('--lstm', action='store_true')
parser.add_argument('--hidden_size', default=128, type=int, help='hidden_size')
parser.add_argument('--n_layers', default=6, type=int, help='num of fc layers')
parser.add_argument('--gamma', default=0.95, type=float, help='decay factor')
parser.add_argument('--history_len', default=5, type=int, help='length of the history used, left zeros')


# training large learning rate can fluctuate! how to prevent fluctuation ? # TODO
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate (default: 0.0001)')
parser.add_argument('--eps', default=1e-5, type=float, help='eps of RMSProp  (default: 1e-5)')
parser.add_argument('--max_grad_norm', default=10, type=float, help='max_grad_norm for clipping grads')
parser.add_argument('--max_grad_value', default=1, type=float, help='max_grad_value for clipping grads')
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--target', default='TD', type=str, help='target = TD/MC, MC only in short episodes (default: TD)')
parser.add_argument('--train_freq', default=5, type=int, help='train every ? frame')
parser.add_argument('--update_freq', default=10, type=int, help='update every ? episode')
# parser.add_argument('--frame_freq', default=3, type=int, help='act every ? frame')

# game
parser.add_argument('--game', default='Breakout-ram-v4', type=str, help='game env name')
parser.add_argument('--disable_byte_norm', action='store_true')
parser.add_argument('--input_rgb', action='store_true')
parser.add_argument('--gray_img', action='store_true')
parser.add_argument('--render', default='rgb_array', type=str, help='where to show? (human/rgb_array)')

#  BeamRider-ram-v4 Breakout-v0  SpaceInvaders-v0  CartPole-v0 BreakoutNoFrameskip-v4 Breakout-v4


args = parser.parse_args()


def main():
    loss_rec = []

    env = gym.make(args.game)
    model_config = args.__dict__
    model_config.update({
        'n_act': env.action_space.n,
        'input_c': env.observation_space.shape[0],
    })
    for k, v in model_config.items():
        logger.info(f'{k}={v}')

    if not os.path.exists('results/'):
        os.mkdir('results/')

    with open(f'results/{args.game}.pickle', 'wb') as handle:
        pickle.dump(model_config, handle)

    agent = DQNAgent(training=True, **args.__dict__)
    score_rec = []
    step = 0
    s_time = time.time()
    for episode in range(args.N_episodes):
        logger.info(f'Epoch={episode}, already finished step={step}')
        obs = env.reset()
        history = deque([obs], maxlen=args.history_len)
        agent.reset()

        # anneal epsilon greedy
        agent.eps_greedy = max(args.eps_greedy, 1 - episode * (1 - args.eps_greedy) / args.explore_step)

        state_dict = {
            'obs': history,
            'la': list(range(env.action_space.n)),
            'reward': None,
            'done': False,
        }
        t, score = 0, 0
        while True:
            step += 1
            env.render(args.render)
            action = agent.step(state_dict)
            obs, reward, done, info = env.step(action)

            history.append(obs)

            state_dict = {
                'obs': history,
                'la': list(range(env.action_space.n)),
                'reward': reward,
                'done': False
            }

            logger.debug(f'Action={action}, R_t+1={reward}')
            score += reward
            t += 1
            if done or t > args.max_len:
                state_dict['done'] = True
                agent.step(state_dict)
                logger.info(f"Episode finished after {t + 1} "
                            f"time steps with reward = {score} "
                            f"remaining={(time.time() - s_time) / (episode + 1) * (args.N_episodes - episode - 1):.3f}s")
                break

            if step % args.train_freq == 0:
                loss = agent.train_loop()
                if loss:
                    loss_rec.append(loss)

        # add to buffer
        score_rec.append(score)
        if args.target == 'TD':
            agent.process_trajectory(final_payoff=reward)
        if args.target == 'MC':
            agent.process_trajectory(final_payoff=score)

        if step % args.update_freq == 0:
            agent.sync_model()
            agent.target_model.save_model(f'{args.game}_v0')

        with open(f'results/{args.game}_reward.pickle', 'wb') as handle:
            pickle.dump(score_rec, handle)
        with open(f'results/{args.game}_loss.pickle', 'wb') as handle:
            pickle.dump(loss_rec, handle)

    plt.plot(score_rec, label='score')
    plt.savefig(f'results/QDN_{args.game}.png')
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stderr, level='INFO')
    logger.add(f'results/DQN_{args.game}.log', level='INFO')
    main()
