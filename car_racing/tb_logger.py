from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    """ Helper class for monitoring and logging results to Tensorboard. """

    def __init__(self):
        self.tb_writer = SummaryWriter()
        self.ep_rewards = []
        self.ep_duration = 0
        self.ep_reward = 0
        self.ep_loss = 0

    def init_episode(self):
        """ Initialize the aggregated episode's metrics  """
        self.ep_rewards.append(self.ep_reward)
        self.ep_duration = 0
        self.ep_reward = 0
        self.ep_loss = 0

    def log_net(self, net, input_to_model):
        """ Depict neural network's architecture to the board in the form of a graph.

         :param net: neural network to be logged
         :param input_to_model: a instance that is going to be fed to the network
         """
        self.tb_writer.add_graph(net, input_to_model)
        self.tb_writer.flush()

    def log_hparams(self, hparams):
        """ Log hyperparameters and evaluation metrics of the current experiment.

         :param hparams: a dictionary with the desired hyperparameters
         """
        metrics = {'episodes needed': len(self.ep_rewards), 'mean reward': sum(self.ep_rewards[-100:]) / 100}
        self.tb_writer.add_hparams(hparams, metrics)
        self.tb_writer.flush()

    def log_step(self, reward, loss):
        """ Update aggregated episode's results with the values of a new step.

         :param reward: the reward value of the step
         :param loss: the loss value of the step
         """
        self.ep_duration += 1
        self.ep_reward += reward
        self.ep_loss += loss

    def log_episode(self, i_ep, eps, steps):
        """ Log episode values to Tensorboard.

        :param i_ep: number of the episode
        :param eps: epsilon value that adjusts exploration rate
        :param steps: total number of steps that have been performed so far
        """

        if self.ep_duration == 0:
            return

        self.tb_writer.add_scalar('Agent/Loss', self.ep_loss / self.ep_duration, i_ep)
        self.tb_writer.add_scalar('Agent/Reward', self.ep_reward, i_ep)
        self.tb_writer.add_scalar('Agent/Epsilon', eps, i_ep)
        self.tb_writer.add_scalar('Agent/Episode_Length', self.ep_duration, i_ep)
        self.tb_writer.add_scalar('Agent/Steps', steps, i_ep)
        self.tb_writer.flush()

        self.init_episode()

    def close(self):
        self.tb_writer.close()
