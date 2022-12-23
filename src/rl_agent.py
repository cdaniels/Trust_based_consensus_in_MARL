
import numpy as np
import gym

default_agent_options = {
    'gamma': 0.9999,
    'epsilon':0.1,
    'alpha': 0.0001,
    'decaying_epsilon': True,
    'epsilon_decay': 0.9996,
    'learning_rate_decay': 1.0
}

class RLAgent:
    def __init__(self, options=default_agent_options, actions=[], states=[], env=None):
        # update options
        self.unpack_options(options)
        # available actions for agent
        self.A = actions
        # states for the enviroment 
        self.S = states
        # data series for plotting
        self.running_average_reward = 0

    def unpack_options(self, options):
        # apply all dictionary key-values as object properties
        for option in options:
            setattr(self, option, options[option])

    def random_argmax(self, array):
        """Return the index of the maximum value from the given array
        randomly selecting from multiple maxes if there is more than one

        Args:
            array (np.array): the array over which to check

        Returns:
            int: the index of the maximum element (or one of the) maximum elements) of the array
        """
        return np.random.choice([i for i, v in enumerate(array) if v == np.max(array)])
        