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

def QA():
    df = pd.read_csv('../processed-data/campaignNetworks_raw.csv')

    print(df.info())

    print(df[df.lname == 'LEWIS']['fname'].unique())

    pass

if __name__ == "__main__":
    #G = readGraph("../processed-data/legislator_bill_edge_list_graph.txt")
    #print(G.GetNodes(), G.GetEdges())

    #G2 = readGraph("../processed-data/campaignNetworks.txt")
    #print "Campaign Graph Nodes: %d, Edges: %d" % (G2.GetNodes(), G2.GetEdges())
    #for NI in G2.Nodes():
    #    if NI.GetId() == 1319:
    #        print "found 1319"

    #CoSponsor = readGraph("../processed-data/cosponsor.txt")
    #print(CoSponsor.GetNodes(), CoSponsor.GetEdges())

    #bill_node = pd.read_csv('processed-data/bill_node.csv')
    #legislator_node = pd.read_csv('processed-data/legislator_node.csv')

    #graphAnalysis(G,bill_node,legislator_node)

    #CoSponsor = getCoSponsor(G,bill_node,legislator_node)
    QA()