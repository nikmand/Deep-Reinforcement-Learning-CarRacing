import torch
import torch.nn as nn


def conv2d_size_out(size, kernel_size, stride):
    """ Helper function to calculate the output dim of conv layer. """
    return (size - (kernel_size - 1) - 1) // stride + 1


class VanillaDQN(nn.Module):
    """ Vanilla implementation of DQN. A single stream is used to estimate the action value function Q(s,a). """

    def __init__(self, output_dim, actions_dim):
        super().__init__()
        self.output = nn.Linear(output_dim, actions_dim)

    def forward(self, x):
        return self.output(x)


class Dueling(nn.Module):
    """ The action value function Q(s,a) can be decomposed to state value V(s) and advantage A(s,a) for each action.
    With Dueling, we want to separate the estimation of V(s) and A(s,a) by using two streams. """

    def __init__(self, output_dim, actions_dim):
        super().__init__()
        # TODO consider refactoring the network arch, streams' layers should be configurable
        self.value_stream = nn.Sequential(nn.Linear(output_dim, 1))

        self.advantage_stream = nn.Sequential(nn.Linear(output_dim, actions_dim))

    def forward(self, x):
        """ The outcome of the first layers of the neural net is passed to both the value and advantage stream.
        Then the results of the two streams are merged and the action value function Q(s,a) is returned. """
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)

        q_values = values + (advantages - advantages.mean())  # broadcasting happens on values

        return q_values


class DQNET(nn.Module):
    """ A convolution neural network inspired by DeepMind's paper: Playing Atari with Deep Reinforcement Learning. """

    def __init__(self, features_dim, fc_layers_dim, actions_dim, dqn_arch=VanillaDQN, dropout=0.1):
        super().__init__()

        c, h, w = features_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_weight = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, kernel_size=8, stride=4), kernel_size=4,
                                                      stride=2), kernel_size=3, stride=1)
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, kernel_size=8, stride=4), kernel_size=4,
                                                      stride=2), kernel_size=3, stride=1)
        conv_output_dim = conv_weight * conv_height * 64

        fc_layers_input_dims = [conv_output_dim] + fc_layers_dim[:-1]
        fc_layers_output_dims = fc_layers_dim
        fc_output_dim = fc_layers_dim[-1]

        self.fc_layers = nn.Sequential(*[nn.Sequential(nn.Linear(input_dim, output_dim),
                                                       # nn.Dropout(p=dropout),  # weird connection in graphs
                                                       nn.ReLU())
                                         for input_dim, output_dim in zip(fc_layers_input_dims, fc_layers_output_dims)])
        self.dqn_arch_layers = dqn_arch(fc_output_dim, actions_dim)

    def forward(self, features):
        conv_outputs = self.conv_layers(features)
        fc_outputs = self.fc_layers(conv_outputs)
        x = self.dqn_arch_layers(fc_outputs)

        return x

    def save_checkpoint(self, filename):
        torch.save(self.state_dict(), filename)

    def load_checkpoint(self, filename):
        weights = torch.load(filename)
        self.load_state_dict(weights)
