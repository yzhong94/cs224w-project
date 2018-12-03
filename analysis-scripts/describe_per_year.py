import snap
import numpy as np
import pandas as pd
from itertools import permutations
from matplotlib import pyplot as plt
import random
import glob


## describe campaign and bill networks by year
## Rule to connect two together: 
### - Campaign data of Year 1 & 2, + Bill Co-Authorship Data of Year 3 & 4
### - In the future, we can try a longer span of campaign records


#### Global var
CAM_YEAR = 2000
BILL_YEAR = 2002 ## read data from processed-data/legislator_bill_edge_list.csv

def readGraph(path):
    G = snap.LoadEdgeList(snap.PUNGraph, path, 0, 1)
    return G

def QA():
    df = pd.read_csv('../processed-data/campaignNetworks_raw_v2.csv')

    print(df.info())

    print(df[df.lname == 'LEWIS']['fname'].unique())

    loadCommMaster()

    pass

def loadCommMaster():
    filePath = "../data/financials/com_master/"
    allFiles = filePath + str(CAM_YEAR) + ".txt"

    result = pd.concat(pd.read_csv(f, sep='|', index_col=False, 
                 names=[
                 'CMTE_ID'
                 ,'CMTE_NM'
                 ,'TRES_NM'
                 ,'CMTE_ST1'
                 ,'CMTE_ST2'
                 ,'CMTE_CITY'
                 ,'CMTE_ST'
                 ,'CMTE_ZIP'
                 ,'CMTE_DSGN'
                 ,'CMTE_TP'
                 ,'CMTE_PTY_AFFILIATION'
                 ,'CMTE_FILING_FREQ'
                 ,'ORG_TP'
                 ,'CONNECTED_ORG_NM'
                 ,'CAND_ID'
                 ]) for f in allFiles)

    #result = result[['CMTE_ID','CMTE_NM', 'CMTE_ST', 'CMTE_TP', 'CMTE_PTY_AFFILIATION', 'CAND_ID']]
    #result = result[['CMTE_ID']]
    #result = result.groupby(['CMTE_ID']).size().reset_index(name='Freq')

    result["ComNodeId"] = result.groupby(['CMTE_ID']).ngroup()
    result.to_csv("../processed-data/com_combined.csv", index = False)
    #print(result.head())

    return

def plotDeg(Graph):
    if Graph == 'campaign':
        G2 = readGraph("../processed-data/campaignNetworks_v2.txt")
    elif Graph == 'bill':
        G2 = readGraph("../processed-data/legislator_bill_edge_list_graph.txt")
    else:
        raise ValueError("Invalid graph: please use 'campaign' or 'bill'. ")

    print Graph + " Graph Nodes: %d, Edges: %d" % (G2.GetNodes(), G2.GetEdges())
    cand = []
    com = []
    for NI in G2.Nodes():
        if NI.GetId() < 10000:
            cand.append(NI.GetOutDeg())
        else:
            com.append(NI.GetOutDeg())
    print Graph + " network max clique size is %d for candidates" % (max(cand))
    print Graph + " network max clique size is %d for non-candidates" % (max(com))

    plt.figure()
    plt.hist(cand, bins=range(min(cand), max(cand)+1), color = "skyblue", ec="skyblue", histtype = 'step', label="Candidates")
    plt.hist(com, bins=range(min(com), max(com)+1), color="red", ec="red", histtype = 'step', label=Graph)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Degree Distribution')
    plt.legend()
    plt.show()

    plt.figure()
    plt.hist(cand, bins=range(min(cand), max(cand)+1), color = "skyblue", ec="blue", histtype = 'step', label="Candidates")
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Degree Distribution for Legislators')
    plt.legend()
    plt.show()

    return

def plotDegOverall():
    G = readGraph("../processed-data/combined_network.txt")

    print " Graph Nodes: %d, Edges: %d" % (G.GetNodes(), G.GetEdges())
    cand = []
    com = []
    bill = []
    for NI in G.Nodes():
        if NI.GetId() < 10000:
            cand.append(NI.GetOutDeg())
        elif NI.GetId() <= 231726: ##com nodeID offset
            bill.append(NI.GetOutDeg())
        else:
            com.append(NI.GetOutDeg())
    print " network highest degree is %d for candidates" % (max(cand))
    print " network highest degree size is %d for bills" % (max(bill))
    print " network highest degree size is %d for committees" % (max(com))

    plt.figure()
    plt.hist(cand, bins=range(min(cand), max(cand)+1), color = "skyblue", ec="blue", histtype = 'step', label="Candidates")
    plt.hist(bill, bins=range(min(com), max(com)+1), color="green", ec="green", histtype = 'step', label="Bills")
    plt.hist(com, bins=range(min(com), max(com)+1), color="red", ec="red", histtype = 'step', label="Committees")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Degree Distribution')
    plt.legend()
    plt.show()

    plt.figure()
    plt.hist(cand, bins=range(min(cand), max(cand)+1), color = "skyblue", ec="blue", histtype = 'step', label="Candidates")
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Degree Distribution')
    plt.legend()
    plt.show()
    return

def projection(Graph):
    if Graph == 'campaign':
        G2 = readGraph("../processed-data/campaignNetworks_v2.txt")
    elif Graph == 'bill':
        G2 = readGraph("../processed-data/legislator_bill_edge_list_graph.txt")
    else:
        raise ValueError("Invalid graph: please use 'campaign' or 'bill'. ")

    H = snap.TUNGraph.New()
    
    for i in G2.Nodes():
        for j in G2.Nodes():
            if (i.GetId() < j.GetId() and j.GetId() < 10000): #10000 is the upper limit for candidate nodes
                NbrV = snap.TIntV()
                Num = snap.GetLen2Paths(G2, i.GetId(), j.GetId(), NbrV)
                if Num > 0:
                    if H.IsNode(i.GetId()) == False:
                        H.AddNode(i.GetId())
                    if H.IsNode(j.GetId()) == False:
                        H.AddNode(j.GetId())
                    if H.IsEdge(i.GetId(), j.GetId()) == False:
                        H.AddEdge(i.GetId(),j.GetId())
    
    print "Compressed Graph Node count total: %d" % (H.GetNodes())

    print "Compressed Edge count total: %d" % (H.GetEdges())
    
    GraphClustCoeff = snap.GetClustCf(H, -1)
    print Graph + " Network Clustering coefficient: %f" % GraphClustCoeff

    snap.SaveEdgeList(H, "../processed-data/"+Graph+"_projection.txt", 
        Graph + " network - Save projected network info as tab-separated list of edges, using unified candidate node IDs")

    return

def getCoSponsor(G, bill_node,legislator_node):
    '''
    returns the one mode projection graph of co-sponsorship

    '''
    CoSponsor = snap.TUNGraph.New()

    print "Legislator count: %d" % len(legislator_node)
    print "Bill count: %d" % len(bill_node)

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

def foldBills():

    G = readGraph("../processed-data/legislator_bill_edge_list_graph.txt")
    bill_node = pd.read_csv('../processed-data/bill_node.csv')
    legislator_node = pd.read_csv('../processed-data/legislator_node.csv')

    H = getCoSponsor(G,bill_node,legislator_node)

    print "Compressed Graph Node count total: %d" % (H.GetNodes())

    print "Compressed Edge count total: %d" % (H.GetEdges())
    
    GraphClustCoeff = snap.GetClustCf(H, -1)
    print "Bill Network Clustering coefficient: %f" % GraphClustCoeff

    snap.SaveEdgeList(H, "../processed-data/bill_projection.txt", 
        "Bill network - Save projected network info as tab-separated list of edges, using unified candidate node IDs")


    return H

def combineGraphs():
    GB = readGraph("../processed-data/legislator_bill_edge_list_graph.txt")
    bill_node = pd.read_csv('../processed-data/bill_node.csv')
    legislator_node = pd.read_csv('../processed-data/legislator_node.csv')

    GC = readGraph("../processed-data/campaignNetworks_v2.txt")
    cnt = 0

    for EI in GB.Edges():
        a = EI.GetSrcNId()
        b = EI.GetDstNId()
        if GC.IsNode(a) == False:
            GC.AddNode(a)
            #print "Adding a legislator node, meaning he/she has no donations, check - node id %d" % (legislator_node['NId'][i])
            if a < 10000:
                cnt = cnt + 1
        if GC.IsNode(b) == False:
            GC.AddNode(b)
            #print "Adding a legislator node, meaning he/she has no donations, check - node id %d" % (legislator_node['NId'][j])
            if b < 10000:
                cnt = cnt + 1
        if GC.IsEdge(a,b) == False:
            GC.AddEdge(a,b)

    print "Added %d new legislator nodes" % (cnt)
    print "Overall graph node count: %d, and edge count %d" % (GC.GetNodes(), GC.GetEdges())

    snap.SaveEdgeList(GC, "../processed-data/combined_network.txt", "Save 1981 to 2016 combined network info as tab-separated list of edges, using unified candidate node IDs")

    return

def statsComm():
    #G = readGraph("../processed-data/combined_network.txt")
    com_node = set()
    GC = readGraph("../processed-data/campaignNetworks_v2.txt") # campaignNetworks_v2
    for NI in GC.Nodes():
        if NI.GetId() >= 10000:
            com_node.add(NI.GetId())

    print "Comm node count %d,  out of %d Nodes" % (len(com_node), GC.GetNodes())
    print "Comm edge count %d" % (GC.GetEdges())

    com_node = set()
    GC = readGraph("../processed-data/legislator_bill_edge_list_graph.txt") # campaignNetworks_v2
    for NI in GC.Nodes():
        if NI.GetId() >= 10000:
            com_node.add(NI.GetId())

    print "Bill node count %d,  out of %d Nodes" % (len(com_node), GC.GetNodes())
    print "Bill edge count %d" % (GC.GetEdges())

    return

if __name__ == "__main__":
    
    ## Functions to generate degree distribution plots
    plotDeg("bill")
    plotDeg("campaign")
    #plotDegOverall()

    ## Functions to fold graphs
    #projection("campaign") ## this will take very long
    #B = foldBills()
    
    ## Functions to pull overall stats for node counts and edge counts
    #combineGraphs()
    #statsComm()