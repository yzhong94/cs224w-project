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
    df = pd.read_csv('../processed-data/campaignNetworks_raw_v2.csv')

    print(df.info())

    print(df[df.lname == 'LEWIS']['fname'].unique())

    loadCommMaster()

    pass

def loadCommMaster():
    filePath = "../data/financials/com_master/"
    allFiles = glob.glob(filePath + "/*.txt")

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

if __name__ == "__main__":
    
    #plotDeg("bill")
    #projection("bill") ## this will take very long
    #plotDeg("campaign")

    projection("campaign") ## this will take very long

