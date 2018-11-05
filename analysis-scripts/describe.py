import snap
import numpy as np
import pandas as pd
from itertools import permutations
from matplotlib import pyplot as plt
import random
import glob

def readGraph(path):
    G = snap.LoadEdgeList(snap.PUNGraph, path, 0, 1)
    return G


if __name__ == "__main__":
    G = readGraph(".../processed-data/legislator_bill_edge_list_graph.txt")
    G2 = readGraph(".../processed-data/campaignNetworks.txt")

    print(G2.GetNodes(), G2.GetEdges())

    #bill_node = pd.read_csv('processed-data/bill_node.csv')
    #legislator_node = pd.read_csv('processed-data/legislator_node.csv')

    #graphAnalysis(G,bill_node,legislator_node)

    #CoSponsor = getCoSponsor(G,bill_node,legislator_node)
    #CoSponsor = readGraph("processed-data/cosponsor.txt")