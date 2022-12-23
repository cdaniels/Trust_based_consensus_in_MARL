
from tabular_learner import TabularLearner, default_agent_options
import numpy as np

class QLearning(TabularLearner):
    def __init__(self, options=default_agent_options, actions=[], states=[]):
        super().__init__(options=options, actions=actions, states=states)

    def policy(self, current_state):
        """Determine an action which best corresponds with the given state

        Args:
            current_state (int): the index fo the current state

        Returns:
            int: the index of the recomended action
        """
        # set current state
        self.current_state = current_state
        # choose A from S using policy derived from Q (e.g, epsilon-greedy)
        action = self.epsilon_greedy(current_state)
        # increment the count for this state and action
        self.N[current_state, action] += 1
        return action

        
    def learn_obs(self, action, next_state, reward):
        """Perform learning on the Q table according to the specified
        action and resulting next state and reward

        Args:
            action (int): the index of the action which just took place
            next_state (int): the index of the resulting state
            reward (int): the reward value
        """
        # Q(S,A) <- Q(S,A) + \alpha*[R + \gamma * max_a(Q(S',a)) - Q(S,A)]
        s = self.current_state
        a, r = action, reward
        alpha_s_a = self.get_alpha_for_state_action(s, a)
        self.Q[s, a] = self.Q[s, a] + alpha_s_a * (r + self.gamma * np.max(self.Q[next_state, :]) - self.Q[s, a])
        # S <- S'
        self.current_state = next_state