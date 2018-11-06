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

    plt.hist(cand)
    plt.title("Candidate Node Degree Distribution in " + Graph + " network")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.show()

    plt.hist(com)
    plt.title("Node Degree Distribution for non-candidates in " + Graph + " network")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.show()
    pass

if __name__ == "__main__":
    
    #plotDeg("bill")

    #plotDeg("campaign")
