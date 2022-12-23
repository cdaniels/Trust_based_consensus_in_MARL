
import numpy as np
import random
import os

from rl_agent import RLAgent, default_agent_options


class TabularLearner(RLAgent):
    def __init__(self, options=default_agent_options, states=[], actions=[], env=None):
        # update options
        self.unpack_options(options)
        # states for the enviroment 
        self.S = states
        self.A = actions
        # state-action values 
        self.V = np.zeros((len(self.S)))
        self.Q = np.zeros((len(self.S), len(self.A)))
        # state-action counts
        self.N = np.zeros((len(self.S), len(self.A)))
        # data series for plotting
        self.running_average_reward = 0
        # policies
        self.pi = np.random.choice(self.A, size=(len(self.S)))
        self.soft_pi = np.random.choice(self.A, size=(len(self.S), len(self.A)))
        # epsilon after decay is applied
        self.effective_epsilon = self.epsilon

    def get_epsilon_for_state(self, state):
        """Get the value for epsilon corresponding with the given 
        state (defaults to a constant if no epsilon-decay is applied)

        Args:
            state (int): the index of some state

        Returns:
            float: a value for epsilon
        """
        epsilon = self.effective_epsilon
        if self.decaying_epsilon:
            # epsilon is chosen based on the visits the state has received
            n_s = (np.sum(self.N[state, :]) + 1) # number of times state s has been visited (epsilon is used to query for the state currently being visited so its always +1)
            # epsilon = 1 / np.sqrt(n_s)
            self.effective_epsilon *= self.epsilon_decay
        return self.effective_epsilon

    def estimate_state_values(self):
        """Estimate values for V based on the Q table

        Returns:
            np.array: an array containing the state values averaged over all actions
        """
        avg_V_for_runs = np.mean(self.Q, axis=2)
        avg_values = np.mean(avg_V_for_runs, axis=0)
        return avg_values

    def epsilon_greedy(self, s: int):
        """Return an action for the given state using a greedy strategy 
        fram the Q-table, occasionally based on a random roll return a
        random action instead (if the roll is below 'epsilon')

        Args:
            s (int): the index for some state

        Returns:
            int: the index of the recomended action
        """
        # choose random number between 0 and 1
        rand_num = np.random.random()
        # if that number is within the exploit range
        if rand_num < self.get_epsilon_for_state(s):
            # simply choose a random action
            return np.random.choice(self.A)
        else:
            # otherwise choose the action with the highest avg reward so far
            return self.random_argmax(self.Q[s])
        
    def get_alpha_for_state_action(self, state, action):
        """Get a learning rate value (alpha) corresponding with
        the given state and action (defaults to returning a constant alpha if no decay is set)

        Args:
            state (int): the index of some state
            action (int): the index of some action

        Returns:
            float: the resulting alpha value
        """
        # alpha is also chosen based on the counts for actions and states
        n_s_a = self.N[state, action] # number of times action a has been taken for state s
        alpha = 1.0
        if n_s_a > 0:
            alpha = 1 / (n_s_a**self.learning_rate_decay)
        return alpha
    
    def get_maximal_action_val(self, state):
        """Return the highest value an action has
        for the given state according to the current Q table

        Args:
            state (int): an index for some state

        Returns:
            float: the value of the highest valued action
        """
        return np.max(self.Q[state, :])