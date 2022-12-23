import numpy as np

from src.environments import NetworkConsensusEnv, default_env_options
from src.q_learning import QLearning, default_agent_options

default_sim_options = {
    'num_episodes': 10,
    'num_runs': 2
}
class Simulation:
    def __init__(self, options=default_sim_options, agent_options=default_agent_options, env=NetworkConsensusEnv(default_env_options)) -> None:
        self.unpack_options(options)
        self.agent_options = agent_options
        self.env = env
        # dictionary for holding agent classes
        self.learner_dict = dict()
        self.per_episode_data = np.zeros((self.num_runs, 2, self.num_episodes))

    def initalize_learners(self, env):
        """Initialize Learning agents for the environment
        constructs on QLearner for each agent

        Args:
            env (Environment): the environment in which learning is taking place
        """
        # clear old learners
        self.learner_dict = dict()
        # initialize agents for environment
        for agent_i in env.agents:
            actions = env.get_actions_for_agent(agent_i)
            states = env.get_states_for_agent(agent_i)
            learner = QLearning(self.agent_options, actions=actions, states=states)
            self.learner_dict[agent_i] = learner

    def initialize_agent_actions(self, agents):
        """Choose initial actions for the given agents

        Args:
            agents (np.arary): a list of agents

        Returns:
            np.array: a list of initial actions
        """
        agent_actions = list()
        for agent_i in agents:
            obs = self.env.get_observation_for_agent(agent_i)
            act = self.learner_dict[agent_i].policy(obs)
            agent_actions.append(act)
        return agent_actions

    def unpack_options(self, options):
        # apply all dictionary key-values as object properties
        for option in options:
            setattr(self, option, options[option])
        
    def run_simulation(self):
        """Perform multiple runs of the simulation and return
        data averaged over the runs

        Returns:
            np.array: an array containing the averaged data
        """
        for i in range(self.num_runs):
            self.perform_run(i)
        return self.get_averaged_per_episode_data()

    def get_averaged_per_episode_data(self):
        # return the per episode data averaged over all the runs
        return np.mean(self.per_episode_data, axis=0)

    def perform_run(self, run):
        """Perform a single run consisting of multiple episodes

        Args:
            run (int): the index of the specific run
        """
        print(f"performing run: {run}")
        self.initalize_learners(self.env)
        episode_count = 0
        while episode_count < self.num_episodes:
            reward_sum, step_count = self.perform_episode()
            # print(f"reward_sum is {reward_sum}")
            # print(f"step_count is {step_count}")

            self.per_episode_data[run][0][episode_count] = reward_sum
            self.per_episode_data[run][1][episode_count] = step_count
            episode_count += 1

    def perform_episode(self):
        """Perform an episode by taking steps in the environment with 
        each agent untill the termination condition is reached

        Returns:
            (int, int): a tuple containing reward sum, and step count for the episode
        """
        # reset the environment
        self.env.reset()
        
        # initialize actions for each agent based on initial states
        agents = self.learner_dict.keys()
        agent_actions = self.initialize_agent_actions(agents)

        terminated = False
        reward_sum, step_count = 0, 0
        while not terminated:
            # broadcast values to neighbors
            for agent_i in agents:
                self.env.broadcast_agent_value(agent_i)

            # receive broadcast values in buffer and update local value from buffer
            for agent_i in agents:
                self.env.receive_incomming_broadcasts(agent_i)

            # perform steps with each agent and learn if appropriate
            for agent_i in agents:
                learner = self.learner_dict[agent_i]

                # take step
                act = agent_actions[agent_i]
                obs, reward, terminated = self.env.step_agent(agent_i, act)
                # increment the counts
                step_count += 1
                reward_sum += reward
                # learn from results
                learner.learn_obs(act, obs, reward)

                # setup the next action for the agent
                agent_actions[agent_i] = learner.policy(obs)

                # termination must break the loop or other agents will reset it
                if(terminated): break
        return reward_sum, step_count