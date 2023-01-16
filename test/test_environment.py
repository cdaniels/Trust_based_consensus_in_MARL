
from src import environments    # The code to test
from src.environments import NetworkConsensusEnv, default_env_options


import unittest   # The test framework
import numpy as np

class Test_Environment(unittest.TestCase):

    def setUp(self) -> None:
        self.env = NetworkConsensusEnv(default_env_options, candidate_values=np.arange(0,50))
        self.agents =self.env.agents
        self.reliable_agents =self.env.reliable_agents
        self.unreliable_agents =self.env.unreliable_agents
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_unpacks_options(self):
        options = {
            'sqrt_num_agents': 3,
            'num_unreliable_agents': 4
        }
        env = NetworkConsensusEnv(options)

        # test if each option is set as an attribute of the envioronment with the same name
        attributes = vars(env)
        for opt in options:
            self.assertIn(opt, attributes)


    def test_initializes_graph_with_nodes_for_each_agent(self):
        sqrt_num_agents = default_env_options['sqrt_num_agents']
        G = self.env.communication_network
        self.assertEqual(sqrt_num_agents**2, len(list(G.nodes())))


    def test_communication_graph_has_nodes_with_ids_matching_agents(self):
        G = self.env.communication_network
        for a in self.agents:
            self.assertIn(a, list(G.nodes()))


    def test_agents_are_divided_into_reliable_and_unreliable(self):
        reliable = self.reliable_agents
        unreliable = self.unreliable_agents
        agents = self.agents
        self.assertEqual(set(agents), set(np.concatenate((reliable, unreliable))))

    def test_reliable_agents_broadcast_proposed_value_to_neighbors(self):
        agent = np.random.choice(self.reliable_agents)
        proposed_value = self.env.get_proposed_value_for_agent(agent)
        self.env.broadcast_agent_value(agent)
        agent_neighbors = self.env.get_agent_neighbors(agent)
        for n in agent_neighbors:
            in_msg = self.env.incomming_messages[n][agent]
            self.assertEqual(in_msg, proposed_value)

    
    def test_unreliable_agents_broadcast_0_to_neighbors(self):
        agent = np.random.choice(self.unreliable_agents)
        self.env.broadcast_agent_value(agent)
        agent_neighbors = self.env.get_agent_neighbors(agent)
        for n in agent_neighbors:
            in_msg = self.env.incomming_messages[n][agent]
            self.assertEqual(in_msg, 0)

    def test_agents_receive_trusted_messages_into_buffer(self):
        self.env.broadcast_proposed_values()
        agent = np.random.choice(self.agents)
        trusted_neighbors = self.env.trusted_neibhors_of_agent(agent)
        in_vals = [self.env.incomming_messages[agent][n] for n in trusted_neighbors]
        buffer = self.env.receive_incomming_messages_into_buffer(agent)
        self.assertEqual(set(in_vals), set(buffer))

    def test_agents_update_proposed_value_from_buffer(self):
        self.env.broadcast_proposed_values()
        agent = np.random.choice(self.agents)
        old_proposed_value = self.env.get_proposed_value_for_agent(agent)
        self.env.broadcast_proposed_values()
        self.env.receive_incomming_broadcasts(agent)
        buffer = self.env.get_message_buffer_for_agent(agent)
        new_proposed_value = self.env.get_proposed_value_for_agent(agent)
        # new values are either old value or value from buffer
        possible_vals = np.concatenate(([old_proposed_value], buffer))
        self.assertIn(new_proposed_value, possible_vals)

    def test_toggle_action_changes_trust_value_for_neighbor(self):
        agent = np.random.choice(self.unreliable_agents)
        self.env.broadcast_agent_value(agent)
        agent_neighbors = self.env.get_agent_neighbors(agent)
        # get the old trust vector
        old_trust_vec = self.env.get_trust_vector_for_agent(agent)
        # neighbor to toggle
        toggle_i = np.random.choice(np.arange(0, len(agent_neighbors)))
        toggle_act = toggle_i + 1 # action is equal to 1 + neighbhor index
        self.env.perform_action(agent, toggle_act)
        # get the new trust vector
        new_trust_vec = self.env.get_trust_vector_for_agent(agent)
        self.assertNotEqual(old_trust_vec[toggle_i], new_trust_vec[toggle_i])

    def test_agent_has_actions_equal_to_neighbors_plus_noop(self):
        agent = np.random.choice(self.agents)
        agent_neighbors = self.env.get_agent_neighbors(agent)
        actions = self.env.get_actions_for_agent(agent)
        self.assertEqual(len(actions), len(agent_neighbors) + 1)

    def test_agent_has_states_equal_to_possible_trust_vectors(self):
        agent = np.random.choice(self.agents)
        trust_vec = self.env.get_trust_vector_for_agent(agent)
        states = self.env.get_states_for_agent(agent)
        num_possible_trust_vec = 2**len(trust_vec)
        self.assertEqual(len(states), num_possible_trust_vec)

    def test_trust_vector_converts_to_state_integer_for_observation(self):
        agent = np.random.choice(self.agents)
        trust_vec = self.env.get_trust_vector_for_agent(agent)
        first_vec = np.zeros((len(trust_vec)))
        last_vec = np.ones((len(trust_vec)))
        num_possible_trust_vec = 2**len(trust_vec)
        first_state_int = self.env.convert_trust_vec_to_state(first_vec)
        last_state_int = self.env.convert_trust_vec_to_state(last_vec)
        self.assertEqual(first_state_int, 0)
        self.assertEqual(last_state_int, num_possible_trust_vec-1)
    
    def test_agent_observes_integer_representing_current_trust_vector(self):
        agent = np.random.choice(self.agents)
        trust_vec = self.env.get_trust_vector_for_agent(agent)
        state_int = self.env.convert_trust_vec_to_state(trust_vec)
        obs, _, _ = self.env.step_agent(agent, 0)
        self.assertEqual(state_int, obs)

    def test_local_consensus_reached_when_agent_agrees_with_neighbors(self):
        # get the proposed value of an agent
        agent = np.random.choice(self.agents)
        proposed_val = self.env.get_proposed_value_for_agent(agent)
        # set all neighbors to have the same proposed value
        agent_neighbors = self.env.get_agent_neighbors(agent)
        for n in agent_neighbors:
            self.env.set_proposed_value_for_agent(n, proposed_val)
        self.assertTrue(self.env.local_consensus_reached(agent))

    def test_local_consensus_not_reached_when_agent_disagrees_with_neighbors(self):
        # get the proposed value of an agent
        agent = np.random.choice(self.reliable_agents)
        proposed_val = self.env.get_proposed_value_for_agent(agent)
        # set all neighbors to have the a different proposed value
        agent_neighbors = self.env.get_agent_neighbors(agent)
        for n in agent_neighbors:
            self.env.set_proposed_value_for_agent(n, proposed_val+1)
        self.assertFalse(self.env.local_consensus_reached(agent))

    def test_global_consensus_reached_when_all_agents_agree_on_1(self):
        # set all agents to have the same value
        proposed_val = 1.0 #np.random.choice(self.env.candidate_values)
        for n in self.agents:
            self.env.set_proposed_value_for_agent(n, proposed_val)
        self.assertTrue(self.env.global_consensus_reached())

    def test_global_consensus_not_reached_when_all_agents_do_not_agree(self):
        # set all agents to have the same value
        proposed_val = np.random.choice(self.env.candidate_values)
        for i, n in enumerate(self.agents):
            self.env.set_proposed_value_for_agent(n, proposed_val+i)
        self.assertFalse(self.env.global_consensus_reached())

    def test_agent_receives_positive_reward_with_local_consensus(self):
        # get the proposed value of an agent
        agent = np.random.choice(self.reliable_agents)
        proposed_val = self.env.get_proposed_value_for_agent(agent)
        # set all neighbors to have the same proposed value
        agent_neighbors = self.env.get_agent_neighbors(agent)
        for n in agent_neighbors:
            self.env.set_proposed_value_for_agent(n, proposed_val)
        self.env.broadcast_proposed_values()
        # take a step with local consensus set
        act = 0 # 0 for noop
        _, reward, _ = self.env.step_agent(agent, act)
        self.assertEqual(reward, 1)

    def test_agent_receives_negative_reward_without_local_consensus(self):
        # get the proposed value of an agent
        agent = np.random.choice(self.reliable_agents)
        proposed_val = self.env.get_proposed_value_for_agent(agent)
        # set all neighbors to have the same different proposed values
        agent_neighbors = self.env.get_agent_neighbors(agent)
        for n in agent_neighbors:
            self.env.set_proposed_value_for_agent(n, proposed_val+1)
        # take a step with local consensus set
        act = 0 # 0 for noop
        _, reward, _ = self.env.step_agent(agent, act)
        self.assertEqual(reward, -1)

    def test_episode_terminates_when_global_consensus_reached(self):
        agent = np.random.choice(self.reliable_agents)
        # set all agents to have the same value
        proposed_val = np.random.choice(self.env.candidate_values)
        for n in self.agents:
            self.env.set_proposed_value_for_agent(n, proposed_val)
        self.env.broadcast_proposed_values()
        # take a step with global consensus reached
        act = 0 # 0 for noop
        _, _, done = self.env.step_agent(agent, act)
        self.assertTrue(done)

if __name__ == '__main__':
    unittest.main()
