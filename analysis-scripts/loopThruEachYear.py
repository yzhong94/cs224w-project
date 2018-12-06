import snap
import numpy as np
import pandas as pd
from itertools import permutations
from matplotlib import pyplot as plt
import random
import glob

CAM_YEAR = 1982
TERM_YEAR = 1984
TERM_START = 98 #1983-1984

## load candidate master to map to nodeIDs
filePath = "../processed-data/candidate_node_mapping_manual.csv"
mapping = pd.read_csv(filePath, index_col=False)

## calculate offset
offset = 0

filePathNode = "../processed-data/"
f = filePathNode + "bill_node.csv"

bill = pd.read_csv(f, delimiter= ",", index_col=False)

offset = bill['NId'].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int64).max()

#print "Committee NodeID Offset is %d" % (offset + 1)

## load bills data
filePathNode = "../processed-data/"
f = filePathNode + "legislator_bill_edge_list.csv"

bills_all = pd.read_csv(f, delimiter= ",", index_col=False)

### MAIN LOOP

while CAM_YEAR < 2016:
    ## load contribution to candidates
	print CAM_YEAR

	filePath = "../data/financials/contributions_to_candidates/"
	allFiles = filePath + str(CAM_YEAR) + ".txt"

	result = pd.read_table(allFiles, sep='|', index_col=False, low_memory=False, 
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
	                 ])

	# process one batch of contribution data

	result = result[['CMTE_ID','TRANSACTION_DT', 'TRANSACTION_AMT', 'CAND_ID']]
	#result = result[['CMTE_ID','CAND_ID']]
	result = result.groupby(['CMTE_ID','TRANSACTION_DT', 'TRANSACTION_AMT', 'CAND_ID']).size().reset_index(name='Freq')
	## remove the case where committees received money from candidates
	result = result[result.TRANSACTION_AMT >= 0]

	result['ContributionYear'] = (result['TRANSACTION_DT'] % 10000).apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int64)
	##remove the case where years are invalid
	result = result[result.ContributionYear > 0]


	result["ComNodeId"] = result.groupby(['CMTE_ID']).ngroup()

	# merge dataframes together
	c = pd.merge(result, mapping, on='CAND_ID', how='left')

	c = c.fillna('')
	c[['NodeID','ComNodeId']] = c[['NodeID','ComNodeId']].apply(pd.to_numeric, errors='coerce').fillna(-1).astype(np.int64)
	    
	## remove entries where Nid = 0, meaning unsuccessful candidates
	c = c[c.fname != '']
	c = c[c.NodeID != -1]

	## add offset
	c['ComNodeId'] = c['ComNodeId'] + offset + 1

	bills = bills_all[bills_all['congress_term'] == TERM_START]

	## Join bill data to campaign contribution data (essentially only include legislators found in the bill data)
	overall = pd.merge(c, bills, left_on='NodeID', right_on = 'DstNId', how='inner')

	print "...Looking at Campaign - Candidate (filtered by bill data) Graph first"
	## NOW, create graphs
	bill_cand = []
	for index, row in bills.iterrows():
	    bill_cand.append(row['DstNId'])
	bill_cand = list(set(bill_cand)) ##dedup

	G_cand = snap.TUNGraph.New()
	for index, row in c.iterrows():
	    if G_cand.IsNode(row['NodeID']) is False and row['NodeID'] in bill_cand:
	        G_cand.AddNode(row['NodeID'])
	    if G_cand.IsNode(row['ComNodeId']) is False and row['NodeID'] in bill_cand:
	        G_cand.AddNode(row['ComNodeId'])
	    if G_cand.IsEdge(row['NodeID'], row['ComNodeId']) is False and row['NodeID'] in bill_cand:
	        G_cand.AddEdge(row['NodeID'], row['ComNodeId'])

	print "G_cand node count is %d" % (G_cand.GetNodes())
	print "G_cand edge count is %d" % (G_cand.GetEdges())

	## Find clique size

	cand = []
	com = []
	#b = []
	for NI in G_cand.Nodes():
	    if NI.GetId() < 10000:
	        cand.append(NI.GetOutDeg())
	    else:
	        com.append(NI.GetOutDeg())
	        
	print "Campaign network max node degree is %d for candidates, out of %d pool" % (max(cand), len(cand))
	print "Campaign network max node degree is %d for committees, out of %d pool" % (max(com), len(com))

	## Fold G_cand to create a one-node projection onto candidate nodes

	H = snap.TUNGraph.New()

	for i in G_cand.Nodes():
	    for j in G_cand.Nodes():
	        if (i.GetId() < j.GetId() and j.GetId() < 10000): #10000 is the upper limit for candidate nodes
	            NbrV = snap.TIntV()
	            Num = snap.GetLen2Paths(G_cand, i.GetId(), j.GetId(), NbrV)
	            if Num > 0:
	                if H.IsNode(i.GetId()) == False:
	                    H.AddNode(i.GetId())
	                if H.IsNode(j.GetId()) == False:
	                    H.AddNode(j.GetId())
	                if H.IsEdge(i.GetId(), j.GetId()) == False:
	                    H.AddEdge(i.GetId(),j.GetId())

	print "One-node Projected Graph Node count total: %d" % (H.GetNodes())

	print "One-node Projected Edge count total: %d" % (H.GetEdges())

	GraphClustCoeff = snap.GetClustCf(H, -1)
	print "Campaign Network Clustering coefficient: %f" % GraphClustCoeff

	MxWcc = snap.GetMxWcc(H)
	print "Campaign Network Max Weakly Connected Component Node Count: %d" % MxWcc.GetNodes()
	print "Campaign Network Max Weakly Connected Component Edge Count: %d" % MxWcc.GetEdges()

	MxScc = snap.GetMxScc(H)
	print "Campaign Network Max Strongly Connected Component Node Count: %d" % MxScc.GetNodes()
	print "Campaign Network Max Strongly Connected Component Edge Count: %d" % MxScc.GetEdges()
	
	####################

	print "...Looking at Bill - Legislator Graph first"
	## NOW, create one-node projection onto legislator graphs

	G_bill = snap.TUNGraph.New()
	for index, row in bills.iterrows():
	    if G_bill.IsNode(row['SrcNId']) is False:
	        G_bill.AddNode(row['SrcNId'])
	    if G_bill.IsNode(row['DstNId']) is False:
	        G_bill.AddNode(row['DstNId'])
	    if G_bill.IsEdge(row['SrcNId'], row['DstNId']) is False:
	        G_bill.AddEdge(row['SrcNId'], row['DstNId'])

	print "G_bill node count is %d" % (G_bill.GetNodes())
	print "G_bill edge count is %d" % (G_bill.GetEdges())

	## Find clique size
	cand = []
	com = []
	#b = []
	for NI in G_bill.Nodes():
	    if NI.GetId() < 10000:
	        cand.append(NI.GetOutDeg())
	    else:
	        com.append(NI.GetOutDeg())
	        
	print "Bill network max node degree is %d for legislators, out of %d pool" % (max(cand), len(cand))
	print "Bill network max node degree is %d for bills, out of %d pool" % (max(com), len(com))

	
	## Fold G_cand to create a one-node projection onto candidate nodes

	H_bill = snap.TUNGraph.New()

	for i in G_bill.Nodes():
	    for j in G_bill.Nodes():
	        if (i.GetId() < j.GetId() and j.GetId() < 10000): #10000 is the upper limit for candidate nodes
	            NbrV = snap.TIntV()
				Num = snap.GetCmnNbrs(G_bill,i.GetId(),j.GetId())
            	#Num = snap.GetLen2Paths(G_bill, i.GetId(), j.GetId(), NbrV)
            	if Num != 0:
	                if H_bill.IsNode(i.GetId()) == False:
	                    H_bill.AddNode(i.GetId())
	                if H_bill.IsNode(j.GetId()) == False:
	                    H_bill.AddNode(j.GetId())
	                if H_bill.IsEdge(i.GetId(), j.GetId()) == False:
	                    H_bill.AddEdge(i.GetId(),j.GetId())

	print "One-node Projected Graph Node count total: %d" % (H_bill.GetNodes())

	print "One-node Projected Edge count total: %d" % (H_bill.GetEdges())

	GraphClustCoeff = snap.GetClustCf(H_bill, -1)
	print "Bill Network Clustering coefficient: %f" % GraphClustCoeff

	MxWcc = snap.GetMxWcc(H_bill)
	print "Bill Network Max Weakly Connected Component Node Count: %d" % MxWcc.GetNodes()
	print "Bill Network Max Weakly Connected Component Edge Count: %d" % MxWcc.GetEdges()

	MxScc = snap.GetMxScc(H_bill)
	print "Bill Network Max Strongly Connected Component Node Count: %d" % MxScc.GetNodes()
	print "Bill Network Max Strongly Connected Component Edge Count: %d" % MxScc.GetEdges()
	

	CAM_YEAR = CAM_YEAR + 2
	TERM_YEAR = TERM_YEAR + 2
	TERM_START = TERM_START + 1
	print "----------------"