import copy
import random
import torch
import math
from nn import DQNET
import logging.config

logging.config.fileConfig('logging.conf')
log = logging.getLogger('car')


class DQNAgent:
    """ Implementation of Deep Q Learning with Fixed Q-targets. This variation uses a second neural net to estimate
    the target Q-value, in order to avoid overestimation. """

    def __init__(self, num_of_actions, network, criterion, optimizer, gamma=0.999, eps_decay=0.0005, eps_start=1,
                 eps_end=0.01, use_gpu=False):
        self.num_of_actions = num_of_actions
        self.gamma = gamma
        self.epsilon = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.policy_net = network.to(self.device)
        self.target_net = self._init_target_net()
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps_decay = eps_decay
        self.eps_start = eps_start
        self.eps_end = eps_end

    def _init_target_net(self):
        """ Create a net of same specifications as policy net and initialize it with the same weights as well. """

        target_net = copy.deepcopy(self.policy_net)
        target_net.load_state_dict(self.policy_net.state_dict())
        target_net.eval()  # gradient updates never happens in target net

        return target_net

    def choose_action(self, state):
        """ Choose an action based on e-greedy if on training mode. """

        if (random.random() < self.epsilon) and self.policy_net.training:
            return random.randrange(self.num_of_actions)
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device)
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was found,
                # so we pick action with the larger expected reward.
                state = state.unsqueeze(0)
                res = self.policy_net(state)
                return res.max(1)[1].item()

    def update(self, transitions):
        """
        :param transitions: a list whose elements are transitions
        """

        states, actions, next_states, rewards, done = list(
            map(lambda tensor: torch.tensor(tensor, device=self.device), zip(*transitions)))

        # Computes Q(s_t, a): the model computes Q(s_t), then we select those columns that correspond to the actions
        # that were originally taken
        predicted_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        # action_batch operates as index, unsqueezed so that each entry corresponds to one row

        # Compute V(s_{t+1}) for all next states.
        next_state_values = self._compute_next_state_values(next_states)
        # Compute the expected Q targets
        target_q_values = rewards + (1 - done.int()) * self.gamma * next_state_values
        # we want to take into account next states' values only if they are not final states
        predicted_q_values = predicted_q_values.squeeze(1)
        target_q_values = target_q_values.detach()

        loss = self.criterion(predicted_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()  # computes gradients
        self.optimizer.step()  # updates weights

        return loss.item()

    def _compute_next_state_values(self, next_states):
        """ next_states batch is passed to the target net. The max value is returned for each entry. """

        next_state_values = self.target_net(next_states).max(1)[0]

        return next_state_values

    def update_target_net(self):
        """ Copies the weights of policy net to target net. """

        self.target_net.load_state_dict(self.policy_net.state_dict())

    def adjust_exploration(self, decaying_schedule):
        """ Progressively decrease the exploration rate. """

        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-decaying_schedule * self.eps_decay)

    def save_checkpoint(self, filename):
        self.policy_net.save_checkpoint(filename)

    def train_mode(self):
        self.policy_net.train()

    def eval_mode(self):
        self.policy_net.eval()


class DDQNAgent(DQNAgent):
    """ Implementation of  Double DQN. The selection of the action is disconnected from the estimation of its value. """

    def __init__(self, num_of_actions, network, criterion, optimizer, gamma=0.99, eps_decay=0.0005, eps_start=1,
                 eps_end=0.01, use_gpu=False):
        super().__init__(num_of_actions, network, criterion, optimizer, gamma, eps_decay, eps_start, eps_end, use_gpu)

    def _compute_next_state_values(self, next_states):
        """ Next_states batch is passed to policy_net and the actions that give the max value are selected.
         Then the batch is passed also to target_net and we return the values that correspond to the actions selected
          by the policy_net. """

        policy_actions = self.policy_net(next_states).max(1)[1]
        next_state_values = self.target_net(next_states).gather(1, policy_actions.unsqueeze(1))

        return next_state_values.squeeze(1)


class DQNAgentBuilder:
    """ Builder takes over the construction of a DQN Agent object. """

    def __init__(self, state_dim, num_of_actions, gamma, eps_decay, eps_start, eps_end, gpu):
        self.network = None
        self.optimizer = None
        self.criterion = None
        self.agent = None
        self.state_dim = state_dim
        self.num_of_actions = num_of_actions
        self.gamma = gamma
        self.eps_decay = eps_decay
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.gpu = gpu

    def build_optimizer(self, optimizer, lr):
        self.optimizer = optimizer(self.network.parameters(), lr)

        return self

    def set_criterion(self, criterion):
        self.criterion = criterion

        return self

    def build_network(self, fc_layers_dims, arch):  # NOTE consider having a builder for Nets
        self.network = DQNET(self.state_dim, fc_layers_dims, self.num_of_actions, arch, dropout=0)

        log.debug("Number of net parameters: {}".format(sum(x.numel() for x in self.network.parameters())))

        return self

    def load_checkpoint(self, checkpoint_path):
        log.info("Loading weights from checkpoint.")

        try:
            self.network.load_checkpoint(checkpoint_path)
        except AttributeError:
            log.warning("No file was provided, checkpointing is aborted.")
        except FileNotFoundError:
            log.warning("File was not found, checkpointing is aborted.")

        return self

    def build(self, agent_algorithm):
        self.agent = agent_algorithm(self.num_of_actions, self.network, self.criterion, self.optimizer, self.gamma,
                                     self.eps_decay, self.eps_start, self.eps_end, self.gpu)
        self.agent.train_mode()

        return self.agent
