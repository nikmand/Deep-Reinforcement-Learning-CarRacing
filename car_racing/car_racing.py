import datetime
import gym
import numpy as np
import torch
import torch.optim as optim
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from tb_logger import TBLogger
from frame_skipping import FrameSkipping
from nn import VanillaDQN, Dueling
from dqn_agent import DQNAgentBuilder, DDQNAgent, DQNAgent
from memory import Memory
from pathlib import Path
import os
import logging.config

logging.config.fileConfig('logging.conf')
log = logging.getLogger('car')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)


if __name__ == '__main__':

    env = gym.make("CarRacing-v0")

    # Transformations are applied to the env based on DeepMind's paper: Playing Atari with Deep Reinforcement Learning
    env = FrameSkipping(env, num_skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    state = env.reset()
    state = np.float32(state.__array__().squeeze())

    # Discreet control is reasonable in this environment as well. Five actions are available.
    discrete_actions = {0: [-1.0, 0.0, 0.0],  # turn left
                        1: [1.0, 0.0, 0.0],  # turn right
                        2: [0.0, 0.0, 0.8],  # break
                        3: [0.0, 0.8, 0],  # accelerate
                        4: [0.0, 0.0, 0.0]  # do nothing
                        }

    # hyperparameters
    lr = 1e-3
    layers_dim = [256]
    gamma = 0.97
    eps_decay, eps_start, eps_end = 0.5 * 1e-4, 1, 0.1
    mem_size = 50_000
    mem_init = 10_000
    max_episodes = 700
    arch = Dueling  # VanillaDQN
    agent_algorithm = DDQNAgent  # DQNAgent
    state_dim = (4, 84, 84)
    num_of_actions = 8
    target_net_upd_period = 1_000  # target net is updated with the weights of policy net every n updates (steps)
    checkpoint_period = max_episodes // 10
    batch_size = 32
    gpu = True

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam

    steps_done = 0  # if continuing from a checkpoint steps_done should be updated accordingly

    agent = DQNAgentBuilder(state_dim, num_of_actions, gamma, eps_decay, eps_start, eps_end, gpu) \
        .set_criterion(criterion) \
        .build_network(layers_dim, arch) \
        .build_optimizer(optimizer, lr) \
        .build(agent_algorithm)

    # the builder can load previously checkpointed weights to the neural network e.g.:
    # .load_checkpoint("checkpoints/2021-09-27T02-13-53/policy_net_107584.pkl")

    memory = Memory(mem_size, batch_size, mem_init)

    tb_logger = TBLogger()

    tb_logger.log_net(agent.policy_net, torch.tensor(state, device=agent.device).unsqueeze(0))

    for i_episode in range(max_episodes):

        log.debug(f"Start of episode: {i_episode}")
        state = env.reset()
        state = np.float32(state.__array__().squeeze())

        done = False

        while not done:
            # env.render()
            action = agent.choose_action(state)
            discrete_action = discrete_actions.get(action)
            next_state, reward, done, _ = env.step(discrete_action)

            next_state = np.float32(next_state.__array__().squeeze())

            memory.store(state, action, next_state, reward, done)  # Store the transition in memory
            state = next_state

            # collect sufficient samples before starting the learning procedure
            if memory.is_initialized():
                transitions = memory.sample()
            else:
                continue

            steps_done += 1
            loss = agent.update(transitions)  # Perform one step of optimization on the policy net

            tb_logger.log_step(reward, loss)

            agent.adjust_exploration(steps_done)  # rate is updated at every step
            if steps_done % target_net_upd_period == 0:
                agent.update_target_net()  # Update the target network

        tb_logger.log_episode(i_episode, agent.epsilon, steps_done)

        if i_episode % checkpoint_period == 0:
            log.info(f"Checkpointing neural network weights at step: {steps_done}")
            save_path = save_dir / f"policy_net_{steps_done}.pkl"
            agent.save_checkpoint(save_path)

    hparams = {'lr': lr, 'gamma': gamma, 'algorithm': str(agent_algorithm), 'arch': str(arch),
               'memory size': mem_size, 'target net upd interval': target_net_upd_period, 'batch size': batch_size,
               'eps decay': eps_decay}

    tb_logger.log_hparams(hparams)

    env.close()
    tb_logger.close()
