import snap
import pandas as pd
import glob
import common_function
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def readGraph(path):
    G = snap.LoadEdgeList(snap.PUNGraph, path, 0, 1)
    return G


def graphAnalysis(G, bill_node, legislator_node):
    '''
    basic function to get some data about the graph for legislators and for bills:
    - clustering coefficient
    - degree distribution
    '''
    print legislator_node.shape
    
    print list(legislator_node)

    legislator_dis = []
    bill_dis = []

    for index, row in legislator_node.iterrows():
        legislator_dis.append(G.GetNI(row['NId']).GetOutDeg())
    
    for index, row in bill_node.iterrows():
        bill_dis.append(G.GetNI(row['NId']).GetOutDeg())

    
    plt.hist(bill_dis,bins = np.arange(0,max(bill_dis),1),histtype = 'step')
    plt.title('Legislator bill network degree distribution plot for bills')
    plt.yscale('log', basey = 10)
    plt.xscale('log', basex = 10)
    plt.show()
    
    plt.hist(legislator_dis,histtype = 'step')
    plt.title('Legislator bill network degree distribution plot for legislators')
    plt.show()
    
    pass

def getCoSponsor(G, bill_node,legislator_node):
    '''
    returns the one mode projection graph of co-sponsorship

    '''
    CoSponsor = snap.TUNGraph.New()

    print len(legislator_node)
    for i in range(len(legislator_node)):
        for j in range(i+1,len(legislator_node)):
            Nbrs = snap.TIntV()
            if snap.GetCmnNbrs(G,legislator_node['NId'][i],legislator_node['NId'][j]) != 0:
                if CoSponsor.IsNode(legislator_node['NId'][i]) == False:
                    CoSponsor.AddNode(legislator_node['NId'][i])
                if CoSponsor.IsNode(legislator_node['NId'][j]) == False:
                    CoSponsor.AddNode(legislator_node['NId'][j])
                if CoSponsor.IsEdge(legislator_node['NId'][i],legislator_node['NId'][j]) == False:
                    CoSponsor.AddEdge(legislator_node['NId'][i],legislator_node['NId'][j]) 

    #snap.SaveEdgeList(CoSponsor, 'cosponsor.txt')

    return CoSponsor

def readToNX(path):
    '''
    taks a txt of edge list and returns a NX
    NOT COMPLETE
    '''
    NX_G = nx.Graph()


    return NX_G
    

def cosponsorGraphAnalysis(CoSponsor):
    
    print "Number of edges -" , CoSponsor.GetEdges()
    print "Number of nodes -" , CoSponsor.GetNodes()
    print "Average clustering coefficient", snap.GetClustCf(CoSponsor)
    
    
    result = snap.GetTriadsAll(CoSponsor)
    print "Number of closed triads", result[0]
    print "Number of open triads", result[2]
    

    pass

if __name__ == "__main__":
    G = readGraph("processed-data/legislator_bill_edge_list_graph.txt")
    bill_node = pd.read_csv('processed-data/bill_node.csv')
    legislator_node = pd.read_csv('processed-data/legislator_node.csv')

    #graphAnalysis(G,bill_node,legislator_node)

    #CoSponsor = getCoSponsor(G,bill_node,legislator_node)
    CoSponsor = readGraph("processed-data/cosponsor.txt")
    cosponsorGraphAnalysis(CoSponsor)

