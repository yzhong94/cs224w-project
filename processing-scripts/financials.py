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
                 #'CAND_OFFICE_ST',
                 'state',
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
	result = result[['CAND_ID','CAND_NAME','state']] 
	result = result.groupby(['CAND_ID','CAND_NAME','state']).size().reset_index(name='Freq')

	result['CAND_NAME'] = map(lambda x: x.upper(), result['CAND_NAME'])
	result['state'] = map(lambda x: x.upper(), result['state'])

	result['fname'], result['lname'] = result['CAND_NAME'].str.split(', ', 1).str
	result['lname'], result['minitial'] = result['lname'].str.split(' ', 1).str

	#result["CanNodeId"] = result.reset_index().index

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

def loadCandidateNodeID():
	filePathNode = "../processed-data/"
	f = filePathNode + "legislator_node.csv"

	canNode = pd.read_csv(f, delimiter= ",", index_col=False 
                 #,names=[
                 #'name',
                 #'bioguide_id',
                 #'state',
                 #'Freq_x',
                 #'thomas_id',
                 #'Freq_y',
                 #'NId']
                 )

	cm = loadCandidateMaster()
	canNode['name'] = map(lambda x: x.upper(), canNode['name'])
	canNode['state'] = map(lambda x: x.upper(), canNode['state'])

	canNode['fname'], canNode['lname'] = canNode['name'].str.split(', ', 1).str
	canNode['lname'], canNode['minitial'] = canNode['lname'].str.split(' ', 1).str

	## Join on fname, lname, and state/CAND_OFFICE_ST
	df = pd.merge(canNode, cm, on=['fname','lname','state'], how='left')

	df = df[['fname','lname', 'name', 'state', 'CAND_ID', 'NId']]

	df = df.groupby(['fname','lname', 'name', 'state', 'CAND_ID', 'NId']).size().reset_index(name='Freq')

	print(df.shape)
	print(df.head())

	return df

def findBillNodeOffset():

	offset = 0

	filePathNode = "../processed-data/"
	f = filePathNode + "bill_node.csv"

	bill = pd.read_csv(f, delimiter= ",", index_col=False)

	offset = bill['NId'].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int64).max()

	return offset

if __name__ == "__main__":

	c = loadContributions()
	can = loadCandidateNodeID() #loadCandidateMaster()
	com = loadCommMaster()

	# merge dataframes together
	c = pd.merge(c, can, on='CAND_ID', how='left')
	c = pd.merge(c, com, on='CMTE_ID', how='left')

	#c = c[['CMTE_ID','TRANSACTION_DT', 'TRANSACTION_AMT', 'CAND_ID', 'CMTE_NM',  'CAND_NAME']]
	c = c.fillna('')

	#print(c.head())
	c[['NId','ComNodeId']] = c[['NId','ComNodeId']].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int64)
	
	## remove entries where Nid = 0, meaning unsuccessful candidates
	c = c[c.fname != '']

	## add differential to commmittee node IDs, based on bill network
	
	d = findBillNodeOffset()
	
	#diff = max(c['NId']) + 1
	#print(diff)

	c['ComNodeId'] = c['ComNodeId'] + d

	print(c.head())

	## create an undirect graph
	G = snap.TUNGraph.New()
	for index, row in c.iterrows():
		if G.IsNode(row['NId']) is False:
			G.AddNode(row['NId'])
		if G.IsNode(row['ComNodeId']) is False:
			G.AddNode(row['ComNodeId'])
		if G.IsEdge(row['NId'], row['ComNodeId']) is False:
			G.AddEdge(row['NId'], row['ComNodeId'])

	print "G node count is %d" % (G.GetNodes())
	print "G edge count is %d" % (G.GetEdges())

	GraphClustCoeff = snap.GetClustCf(G, -1)
	print "Clustering coefficient: %f" % (GraphClustCoeff) ## TODO: debug, this value = 0, seems wrong

	x, y = getDataPointsToPlot(G)
	plt.loglog(x, y, linestyle = 'dotted', color = 'b', label = '2008 - 2016 Campaign Financial Network')
	plt.xlabel('Node Degree (log)')
	plt.ylabel('Proportion of Nodes with a Given Degree (log)')
	plt.title('Degree Distribution of Campaign Financial Network')
	plt.legend()
	plt.show()

	snap.SaveEdgeList(G, "../processed-data/campaignNetworks.txt", "Save 2008 to 2016 campaign network info as tab-separated list of edges, using unified candidate node IDs")