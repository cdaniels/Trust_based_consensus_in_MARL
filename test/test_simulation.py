
from src import simulation    # The code to test
from src.simulation import Simulation
from src.environments import SARGridWorld, default_options


import unittest   # The test framework
import numpy as np

class Test_Simulation(unittest.TestCase):

    def setUp(self) -> None:
        self.env = SARGridWorld(default_options)
        self.sim = Simulation(env=self.env)
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_simulation_initializes_learners(self):
        self.assertTrue(False)

    def test_simulation_performs_step_with_each_agent(self):
        self.assertTrue(False)

    def simulation_terminates_if_any_step_terminates(self):
        self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()
