import numpy as np
import networkx as nx

import itertools as it
from enum import Enum


default_env_options = {
    # 2 means 4 agents total ,3 means 9 agents total, etc...
    'sqrt_num_agents': 3,
    'num_unreliable_agents': 4
}
class NetworkConsensusEnv:

    def __init__(self, options, candidate_values=[0,1]) -> None:
        # overidable defaults
        self.sqrt_num_agents = 3
        # unpack the options
        self.unpack_options(options)
        self.num_agents = self.sqrt_num_agents**2
        # graph representing communication network for agents
        self.communication_network = nx.convert_node_labels_to_integers(nx.grid_2d_graph(self.sqrt_num_agents, self.sqrt_num_agents))
        # agents divided into reliable and unreliable
        self.agents = np.array(list(self.communication_network.nodes()))
        self.unreliable_agents = np.random.choice(self.agents, self.num_unreliable_agents, replace=False)
        self.reliable_agents = self.agents[np.isin(self.agents, self.unreliable_agents, invert=True)]
        # agent proposed values from candidate values
        self.candidate_values = candidate_values
        # conversion table for agent neighbhor trust state
        self.state_conversions = self.generate_state_conversions()
        self.reset()

    def __del__(self):
        pass

    def reset(self):
        """Resut all communication, trust, and proposed value information for all agents
        """
        self.proposed_values = np.random.choice(self.candidate_values, self.num_agents) # random inital guesses for value by each agent
        # tables representing incomming broadcasts between agents and trust between agents
        self.incomming_messages = np.ones((self.num_agents, self.num_agents)) * -1
        self.trust_matrix = np.ones((self.num_agents, self.num_agents))
        # buffer for holding incomming messages from neighbhors
        self.message_buffers = dict()

    def generate_state_conversions(self):
        """Generate a nested dictionary containing state converrsions
        the first level keys are integers 2, 3, 4 representing the possible numbers of neighbors an agent can have
        the first level values are themselves dictionaries containing mappings for trust vectors of the specified size

        For the second dictionary
        the second level keys are tuples representing possible trust states for the trust vectors corresponding with
        that number of neighbors, and the second values are integers which uniquely correspond with these tuples

        Returns:
            dict: the conversion dict
        """
        # create state conversion tables for corner node(2), sides node(3), and central node neibhors(4)
        state_conversions = dict()
        for num_states in [2, 3, 4]:
            possible_states = list(it.product([0,1], repeat=num_states))
            flipped_states = { a: i for i, a in enumerate(possible_states)}
            state_conversions[num_states] = flipped_states
        return state_conversions

    def get_agent_neighbors(self, agent_i: int):
        return sorted([n for n in self.communication_network.neighbors(agent_i)])

    def get_proposed_value_for_agent(self, agent_i: int):
        return self.proposed_values[agent_i]

    def set_proposed_value_for_agent(self, agent_i: int, val: int):
        self.proposed_values[agent_i] = val

    def get_message_buffer_for_agent(self, agent_i: int):
        return self.message_buffers[agent_i]

    def get_trust_vector_for_agent(self, agent_i: int):
        """Get a vector representing the given agent's trust in its neighboring
        agents whose elemnts are either 1 for trusted, or 0 for untrusted

        Args:
            agent_i (int): the index of the agent in question

        Returns:
            np.array: the trust vector
        """
        nei_i = self.get_agent_neighbors(agent_i)
        trust_vector = np.ones(len(nei_i), dtype='int')
        for i, n in enumerate(nei_i):
            if not self.agent_trusts_agent(agent_i, n):
                trust_vector[i] = 0
        return trust_vector

    def local_consensus_reached(self, agent_i: int):
        """ Determine whether or not a certain agent
        has achieved local consensus with its immediate neibhors
        (ie: whether its proposed values agree with those of its neighbors (as value=1))

        Args:
            agent_i (int): the index of the agent

        Returns:
            bool: whether or not a local consensus was reached for that agent
        """
        nei_i = self.get_agent_neighbors(agent_i)
        agent_val = self.get_proposed_value_for_agent(agent_i)
        for n in nei_i:
            neighbor_val = self.get_proposed_value_for_agent(n)
            if n not in self.unreliable_agents and neighbor_val != agent_val:
                return False
        return True

    def global_consensus_reached(self):
        """ Determine whether or not all agents
        has achieved consensus (ie: whether their probosed vaules agree (as value=1))

        Returns:
            bool: whether or not a global consensus was reached for all agents
        """
        for a in self.agents:
            agent_val = self.get_proposed_value_for_agent(a)
            for b in self.agents:
                other_val = self.get_proposed_value_for_agent(b)
                if a not in self.unreliable_agents and b not in self.unreliable_agents and agent_val != other_val:
                    return False
        return True

    def broadcast_proposed_values(self):
        """ Make all agents broadcast their proposed values to their neighbors
        """
        for agent_i in self.agents:
            self.broadcast_agent_value(agent_i)

    def broadcast_agent_value(self, agent_i: int):
        """Broadcast the given agent's proposed value to all its neighbor agents
        (these values will fill the 'incoming_messages' table)

        Args:
            agent_i (int): the index of the agent in question
        """
        # get current value v_i
        v_i = self.proposed_values[agent_i]
        # broadcast v_i to neibhoring nodes
        nei_i = self.get_agent_neighbors(agent_i)
        for n in nei_i:
            # if agent is reliable broadcast the proposed value
            if agent_i in self.reliable_agents:
                self.incomming_messages[n][agent_i] = v_i
            # otherwise broadcast 0
            else:
                self.incomming_messages[n][agent_i] = 0

    def receive_incomming_messages_into_buffer(self, agent_i):
        """ Take all incomming messages for the specifed agent and read them into a buffer

        Args:
            agent_i (int): the index of the agent in question

        Returns:
            np.array: an array containing all incoming messages for the agent
        """
        # receive incoming messages from trusted agents into buffer
        trusted_nei_i = self.trusted_neibhors_of_agent(agent_i)
        buffer_i = np.array([self.incomming_messages[agent_i][n] for n in trusted_nei_i])
        self.message_buffers[agent_i] = buffer_i
        return buffer_i

    def trusted_neibhors_of_agent(self, agent_i: int):
        """Return the trusted neighbors of the given agent

        Args:
            agent_i (int): the index of the agent in question

        Returns:
            np.array: an array containing the node ids of the trusted neighbhors of the agent
        """
        nei_i = self.get_agent_neighbors(agent_i)
        trusted_nei_i = list()
        for n in nei_i:
            if self.agent_trusts_agent(agent_i, n):
                trusted_nei_i.append(n)
        return np.array(trusted_nei_i)
        
    def agent_trusts_agent(self, agent_i: int, agent_j: int):
        """Trust is defined as either a 1 or a 0

        Args:
            agent_i (int): agent who trusts
            agent_j (int): trusted agent

        Returns:
            bool: whether or not agent_i trusts agent_j
        """
        return self.trust_matrix[agent_i][agent_j] == 1
    
    def update_agent_value(self, agent_i: int, buffer_i: np.array):
        """Update the proposed value for an agent by selecting from
        its current value and the buffer containing incoming messages

        Args:
            agent_i (int): the agent to update
            buffer_i (np.array): the buffer of received messages
        """
        # get current value v_i
        v_i = self.proposed_values[agent_i]
        # update current value by randomly selecting from {v_i} union buffer_i
        next_v_i = np.random.choice(np.union1d([v_i], buffer_i))
        self.proposed_values[agent_i] = next_v_i
 
    def unpack_options(self, options):
        """ Assign given options as class properties

        Args:
            options (dict): a dictionary containing option parameters
        """
        # apply all dictionary key-values as object properties
        for option in options:
            setattr(self, option, options[option])

    def toggle_trust(self, agent_i: int, toggle_i: int):
        """Toggle the trust vale for the specified neighbor of the agent

        Args:
            agent_i (int): the index of the agent in question
            toggle_i (int): an index of one of the agent's neighbors
        """
        neighbor_node_i = self.get_node_index_for_agent_neighbor(agent_i, toggle_i)
        old_val = self.trust_matrix[agent_i][neighbor_node_i]
        new_val = int(not old_val)
        self.trust_matrix[agent_i][neighbor_node_i]  = new_val

    # def get_agent_neighbor_index_from_node(self, agent_i: int, node_i: int):

    def get_node_index_for_agent_neighbor(self, agent_i: int, neighbhor_j: int):
        """ Get the index for the node (from the communication network graph) which 
        corresponds with the given agents neibhor index

        Args:
            agent_i (int): the agent in question
            neighbhor_j (int): an index for one of the agents neighbors (ie; 3 neighber --> 0,1,2)

        Raises:
            Exception: incorrect index passed

        Returns:
            int: the id for the node (from the communictaion network graph)
        """
        nei_i = self.get_agent_neighbors(agent_i)
        for i, n in enumerate(nei_i):
            if i == neighbhor_j:
                return n
        raise Exception("incorrect neighbor index for agent")

    def perform_action(self, agent_i: int, action: int):
        """Perfom the specified action for the given agent
        actions are either a trust togle or no action at all (noop)

        Args:
            agent_i (int): the index of the agent in question
            action (int): the action index
        """
        if action != 0:
            # toggle the specified neibhors trust value
            self.toggle_trust(agent_i, action-1)
        else:
            # otherwise do nothing
            pass

    def reward_for_agent(self, agent_i: int):
        """ Get the reward for the given agent based on its consensus

        Args:
            agent_i (int): the index of the agent in question

        Returns:
            int: the reward value
        """
        reward = 0.0
        if self.local_consensus_reached(agent_i):
            reward = 1.0
        else:
            reward = -1.0
        return reward

    def receive_incomming_broadcasts(self, agent_i):
        """Recieve incomming messages for the given agent into a buffer
        and trigger an update of the agent's proposed values based on them
        

        Args:
            agent_i (int): the index of the agent to receive messages
        """
        buffer_i = self.receive_incomming_messages_into_buffer(agent_i)
        self.update_agent_value(agent_i, buffer_i)
    
    def step_agent(self, agent_i, action):
        # self.receive_communications(agent_i)
        # perform the specified action
        self.perform_action(agent_i, action)
        # get the observation, reward, and termination condition and return them as a tuple
        obs = self.get_observation_for_agent(agent_i)
        reward = self.reward_for_agent(agent_i)
        done = self.global_consensus_reached()
        return (obs, reward, done)

    def get_observation_for_agent(self, agent_i):
        """ get the observation of the environment for any given agent
        Args:
            agent_i (_type_): _description_
        Returns:
            _type_: _description_
        """
        trust_vec = self.get_trust_vector_for_agent(agent_i)
        state_int = self.convert_trust_vec_to_state(trust_vec)
        return state_int


    def convert_trust_vec_to_state(self, trust_vec: np.array):
        """Convert the given trust vector into a unique integer
        using the conversion table

        Args:
            trust_vec (np.array): the trust vector

        Returns:
            int: a unique integer representing the specific configuration of the vectors entries
        """
        num_states = len(trust_vec)
        return self.state_conversions[num_states][tuple(trust_vec)]
    
    def get_actions_for_agent(self, agent_i: int):
        noop_action = np.array([0])
        toggle_actions = np.array([i+1 for i, n in enumerate(self.get_agent_neighbors(agent_i))])
        return np.concatenate((noop_action, toggle_actions))

    def get_states_for_agent(self, agent_i: int):
        num_states = len(self.get_trust_vector_for_agent(agent_i))
        return self.state_conversions[num_states].values()
        