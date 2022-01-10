
# python agents/human.py
# python dqn_trainer.py --N_episodes 10000 --S cp --disable_byte_norm --human --game CartPole-v0
# python test.py --C cppg --A pg


import random
import torch
from loguru import logger
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch import FloatTensor, LongTensor
from loguru import logger
from agents.pg import PGAgent
import gym

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

frame_freq = 1
history_len = 5

env = gym.make('Breakout-v0')  # 'SpaceInvaders-v0'
target = 'TD'
# env = gym.make('Breakout-v0')
agent = PGAgent(critic='q',
                critic_target='mc',
                n_act=env.action_space.n,
                history_len=history_len,
                input_c=env.observation_space.shape[0])

obs = env.reset()
history = [obs]
# traj = [agent.process_vec_obs(history)]
state_dict = {
    'obs': history,
    'la': list(range(env.action_space.n)),
    'reward': None,
    'done': False,
}
t, score, frame = 0, 0, 0
act = None
while True:
    env.render()
    if frame <= 0:
        act = agent.act(state_dict)
        frame = frame_freq - 1
        # traj.append(action)
    obs, reward, done, info = env.step(act)
    print(reward)
    history.append(obs)
    if len(history) > history_len:
        history.pop(0)

    state_dict = {
        'obs': history,
        'la': list(range(env.action_space.n)),
        'reward': reward,
        'done': False
    }
    # traj.append(reward)
    # traj.append(agent.process_vec_obs(history))

    logger.debug(f'Action={act}, R_t+1={reward}')
    t += 1
    frame -= 1
    score += reward
    if done or t > 1000:
        state_dict['done'] = True
        agent.act(state_dict)
        logger.info(f"Episode finished after {t} time steps, total reward={score}")
        break

agent.backup()




