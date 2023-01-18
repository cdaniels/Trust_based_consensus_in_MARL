
from tabular_learner import TabularLearner, default_agent_options
import numpy as np

class DoubleQLearning(TabularLearner):
    def __init__(self, options=default_agent_options, actions=[], states=[]):
        super().__init__(options=options, actions=actions, states=states)
        self.Q1 = np.zeros((len(self.S), len(self.A)))
        self.Q2 = np.zeros((len(self.S), len(self.A)))

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
        s = self.current_state
        a, r = action, reward
        
        # update Q
        self.Q = np.add(self.Q1, self.Q2)

        # get state-action dependent alpha
        alpha_s_a = self.get_alpha_for_state_action(s, a)

        # with 0.5 probability
        rand_num = np.random.random()
        if rand_num < 0.5:
            # Q1(S,A) <- Q1(S,A) + \alpha*[R + \gamma * Q2[S', max_a(Q1(S',a))] - Q1(S,A)]
            self.Q1[s, a] = self.Q1[s, a] + alpha_s_a * (r + self.gamma * self.Q2[next_state, self.random_argmax(self.Q1[next_state, :])] - self.Q1[s, a])
        else:
            # Q2(S,A) <- Q2(S,A) + \alpha*[R + \gamma * Q1[S', max_a(Q2(S',a))] - Q2(S,A)]
            self.Q2[s, a] = self.Q2[s, a] + alpha_s_a * (r + self.gamma * self.Q1[next_state, self.random_argmax(self.Q2[next_state, :])] - self.Q2[s, a])
        
        
        
        

        # S <- S'
        self.current_state = next_state


    def perform_episode(self, run):
        # initalize S for new episode
        initial_state = self.env.initialize()
        # iterate steps of episode untill termination
        terminated = False
        sum_of_rewards = 0
        steps = 0
        while not terminated:
            # S <- S'
            initial_state = next_state

            # update data
            sum_of_rewards += reward
            steps += 1
        # update policy
        self.update_active_policy(run)
        return sum_of_rewards