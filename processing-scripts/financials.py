import snap
import numpy as np
import pandas as pd
from itertools import permutations
from matplotlib import pyplot as plt
import random
import glob


def loadCandidateMaster():
	filePath = "../data/financials/candidate_master/"
	allFiles = glob.glob(filePath + "/*.txt")

	result = pd.concat(pd.read_csv(f, sep='|', index_col=False, 
                 names=['CAND_ID',
                 'CAND_NAME',
                 'CAND_PTY_AFFILIATION',
                 'CAND_ELECTION_YR',
                 'CAND_OFFICE_ST',
                 'CAND_OFFICE',
                 'CAND_OFFICE_DISTRICT',
                 'CAND_ICI',
                 'CAND_STATUS',
                 'CAND_PCC',
                 'CAND_ST1',
                 'CAND_ST2',
                 'CAND_CITY',
                 'CAND_ST',
                 'CAND_ZIP']) for f in allFiles)
	
	#result = result[['CAND_ID','CAND_NAME', 'CAND_ELECTION_YR', 'CAND_OFFICE_ST', 'CAND_OFFICE']]
	result = result[['CAND_ID','CAND_NAME']] 
	result = result.groupby(['CAND_ID','CAND_NAME']).size().reset_index(name='Freq')

	result["CanNodeId"] = result.reset_index().index

	#print(result.head())

	return result

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
	result = result[['CMTE_ID']]
	result = result.groupby(['CMTE_ID']).size().reset_index(name='Freq')

	result["ComNodeId"] = result.reset_index().index

	#print(result.head())

	return result

def loadContributions():
	filePath = "../data/financials/contributions_to_candidates/"
	allFiles = glob.glob(filePath + "*.txt")

	result = pd.concat(pd.read_table(f, sep='|', index_col=False, low_memory=False, 
                 names=[
                 'CMTE_ID'
                 ,'AMNDT_IND'
                 ,'RPT_TP'
                 ,'TRANSACTION_PGI'
                 ,'IMAGE_NUM'
                 ,'TRANSACTION_TP'
                 ,'ENTITY_TP'
                 ,'NAME'
                 ,'CITY'
                 ,'STATE'
                 ,'ZIP_CODE'
                 ,'EMPLOYER'
                 ,'OCCUPATION'
                 ,'TRANSACTION_DT'
                 ,'TRANSACTION_AMT'
                 ,'OTHER_ID'
                 ,'CAND_ID'
                 ,'TRAN_ID'
                 ,'FILE_NUM'
                 ,'MEMO_CD'
                 ,'MEMO_TEXT'
                 ,'SUB_ID'
                 ]) for f in allFiles)

	#result = result[['CMTE_ID','TRANSACTION_DT', 'TRANSACTION_AMT', 'CAND_ID']]
	result = result[['CMTE_ID','CAND_ID']]
	result = result.groupby(['CMTE_ID','CAND_ID']).size().reset_index(name='Freq')

	return result
def getDataPointsToPlot(Graph):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return values:
    X: list of degrees
    Y: list of frequencies: Y[i] = fraction of nodes with degree X[i]
    """
    ############################################################################
    # TODO: Your code here!
    N = Graph.GetNodes()
    DegToCntV = snap.TIntPrV()
    snap.GetDegCnt(Graph, DegToCntV)
    X, Y = [], []
    for item in DegToCntV:
    	#print "%d nodes with degree %d" % (item.GetVal2(), item.GetVal1())
    	X.append(item.GetVal1())
    	Y.append(item.GetVal2()*1.0/N)
    ############################################################################
    return X, Y

if __name__ == "__main__":

	c = loadContributions()
	can = loadCandidateMaster()
	com = loadCommMaster()

	# merge dataframes together
	c = pd.merge(c, can, on='CAND_ID', how='left')
	c = pd.merge(c, com, on='CMTE_ID', how='left')

	#c = c[['CMTE_ID','TRANSACTION_DT', 'TRANSACTION_AMT', 'CAND_ID', 'CMTE_NM',  'CAND_NAME']]
	c = c.fillna('')

	#print(c.head())
	c[['CanNodeId','ComNodeId']] = c[['CanNodeId','ComNodeId']].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int64)
	
	diff = max(c['CanNodeId']) + 1
	c['ComNodeId'] = c['ComNodeId'] + diff

	print(c.head())

	G = snap.TUNGraph.New()
	for index, row in c.iterrows():
		if G.IsNode(row['CanNodeId']) is False:
			G.AddNode(row['CanNodeId'])
		if G.IsNode(row['ComNodeId']) is False:
			G.AddNode(row['ComNodeId'])
		if G.IsEdge(row['CanNodeId'], row['ComNodeId']) is False:
			G.AddEdge(row['CanNodeId'], row['ComNodeId'])

	print "G node count is %d" % (G.GetNodes())
	print "G edge count is %d" % (G.GetEdges())

	GraphClustCoeff = snap.GetClustCf(G, -1)
	print "Clustering coefficient: %f" % (GraphClustCoeff)

	x, y = getDataPointsToPlot(G)
	plt.loglog(x, y, linestyle = 'dotted', color = 'b', label = '2008 - 2016 Campaign Financial Network')
	plt.xlabel('Node Degree (log)')
	plt.ylabel('Proportion of Nodes with a Given Degree (log)')
	plt.title('Degree Distribution of Campaign Financial Network')
	plt.legend()
	plt.show()

	snap.SaveEdgeList(G, "../data/campaignNetworks.txt", "Save 2008 to 2016 campaign network info as tab-separated list of edges")
	##TODO: use bb's node IDs