import gym
import sys
import time
import pickle
import os
import torch
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
parser.add_argument('--S', default='run', help='name for the experiment')
parser.add_argument('--load_ckp', action='store_true')
# buffer
parser.add_argument('--N_episodes', default=100000, type=int, help='N_episodes')
parser.add_argument('--max_len', default=100000, type=int, help='max_len of episodes')
parser.add_argument('--buffer_size', default=10000, type=int, help='buffer_size of trajectory')
parser.add_argument('--eps_greedy', default=0.05, type=float, help='eps_greedy (default: 0.1)')
parser.add_argument('--explore_step', default=500, type=int, help='anneal greedy')

# model  lstm can easily blow up
parser.add_argument('--lstm', action='store_true')
parser.add_argument('--momentum', default=0.1, type=float, help='momentum for BatchNorm')
parser.add_argument('--hidden_size', default=128, type=int, help='hidden_size')
parser.add_argument('--n_layers', default=6, type=int, help='num of fc layers')
parser.add_argument('--gamma', default=0.99, type=float, help='decay factor')
parser.add_argument('--history_len', default=5, type=int, help='length of the history used, left zeros')


# training large learning rate can fluctuate! how to prevent fluctuation ? # TODO
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate (default: 0.0001)')
parser.add_argument('--eps', default=1e-5, type=float, help='eps of RMSProp  (default: 1e-5)')
parser.add_argument('--max_grad_norm', default=10, type=float, help='max_grad_norm for clipping grads')
parser.add_argument('--max_grad_value', default=1, type=float, help='max_grad_value for clipping grads')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--target', default='TD', type=str, help='target = TD/MC, MC only in short episodes (default: TD)')
parser.add_argument('--train_freq', default=5, type=int, help='train every ? frame')
parser.add_argument('--update_freq', default=20, type=int, help='update every ? EPISODE')
# parser.add_argument('--frame_freq', default=3, type=int, help='act every ? frame')

# game
parser.add_argument('--game', default='Breakout-v4', type=str, help='game env name')
parser.add_argument('--disable_byte_norm', action='store_true')
parser.add_argument('--input_rgb', action='store_true')
parser.add_argument('--render', default='rgb_array', type=str, help='where to show? (human/rgb_array)')

#  BeamRider-ram-v4 Breakout-v0  SpaceInvaders-v0  CartPole-v0 BreakoutNoFrameskip-v4 Breakout-v4


args = parser.parse_args()


def main():
    if args.load_ckp:
        with open(f'checkpoints/{args.S}_ckp.pickle', 'rb') as handle:
            checkpoint = pickle.load(handle)
        config = checkpoint['config']

        logger.info('Loading checkpoints...')

        for k, v in args.__dict__.items():
            if config[k] != v and k != 'load_ckp':
                logger.info(f'update>> {k}: {config[k]} -> {v}')
                config[k] = v

        for k, v in config.items():
            logger.info(f'{k}={v}')
        env = gym.make(config['game'])
    else:
        checkpoint = {'episode': 0,
                      'config': {},
                      'reward': [],
                      'loss': [],
                      'model_dict': None}
        config = args.__dict__
        env = gym.make(args.game)
        config.update(n_act=env.action_space.n,
                      input_c=env.observation_space.shape[0])
        for k, v in config.items():
            logger.info(f'{k}={v}')

        if not os.path.exists('checkpoints/'):
            os.mkdir('checkpoints/')

        # with open(f'results/{args.game}.pickle', 'wb') as handle:
        #     pickle.dump(config, handle)
        checkpoint.update(config=config)

    agent = DQNAgent(training=True, **config)
    if checkpoint['model_dict']:
        agent.policy_model.load_state_dict(checkpoint['model_dict']['policy'])
        agent.target_model.load_state_dict(checkpoint['model_dict']['target'])

    score_rec = checkpoint['reward']
    loss_rec = checkpoint['loss']
    loss = None
    step = 0
    la = list(range(env.action_space.n))
    s_time = time.time()

    min_episode = checkpoint['episode']

    for episode in range(min_episode, config['N_episodes']):
        logger.info(f'Epoch={episode}, already finished step={step}')
        obs = env.reset()
        history = deque([obs], maxlen=config['history_len'])
        agent.reset()

        # anneal epsilon greedy
        agent.eps_greedy = config['eps_greedy']
        # agent.eps_greedy = max(config['eps_greedy'],
        #                        1 - episode * (1 - config['eps_greedy']) / config['explore_step'])

        state_dict = {
            'obs': history,
            'la': la,
            'reward': None,
            'done': False,
        }
        t, score = 0, 0
        while True:
            step += 1
            env.render(config['render'])
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

            # training
            if step % args.train_freq == 0:
                current_loss = agent.train_loop()
                if current_loss:
                    loss = current_loss

            # end of this episode
            if done or t > config['max_len']:
                state_dict['done'] = True
                agent.step(state_dict)
                logger.info(f"Episode finished after {t + 1} "
                            f"time steps with reward = {score} "
                            f"remaining={(time.time() - s_time) / (episode + 1) * (config['N_episodes'] - episode - 1):.3f}s")
                break

        if config['target'] == 'TD':
            agent.process_trajectory(final_payoff=reward)
        if config['target'] == 'MC':
            agent.process_trajectory(final_payoff=score)

        # update policy model as the target
        if episode % config['update_freq'] == 0:
            agent.sync_model()
            # agent.target_model.save_model(f'{config["game"]}_{config["S"]}')
            if loss:
                loss_rec.append(loss)
            score_rec.append(score)

            checkpoint.update(model_dict={'policy': agent.policy_model.state_dict(),
                                          'target': agent.target_model.state_dict()})
            checkpoint.update(episode=episode)
            checkpoint.update(reward=score_rec)
            checkpoint.update(loss=loss_rec)

        with open(f'checkpoints/{config["S"]}_ckp.pickle', 'wb') as handle:
            pickle.dump(checkpoint, handle)

    plt.plot(score_rec, label='score')
    plt.savefig(f'results/QDN_{config["game"]}.png')
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stderr, level='INFO')
    logger.add(f'results/DQN_{args.game}.log', level='INFO')
    main()
