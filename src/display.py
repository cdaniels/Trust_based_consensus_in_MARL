
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from environments import NetworkConsensusEnv


class Display:
    def __init__(self) -> None:
        pass

    
    def visit(self, env:NetworkConsensusEnv):
        g = env.communication_network
        trust = env.trust_matrix
        nx.draw(g)
        print(trust)
        plt.show()
        