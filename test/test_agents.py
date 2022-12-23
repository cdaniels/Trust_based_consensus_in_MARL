
from src import q_learning    # The code to test
from src.q_learning import QLearning, default_agent_options
from src.environments import NetworkConsensusEnv, default_env_options


import unittest   # The test framework
import numpy as np

class Test_QLearning(unittest.TestCase):

    def setUp(self) -> None:
        self.env = NetworkConsensusEnv(default_env_options)
        self.agent = QLearning(default_agent_options)
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()
