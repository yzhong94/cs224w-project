import snap
import numpy as np
import pandas as pd
from itertools import permutations
from matplotlib import pyplot as plt
import random
import glob

## Create campaign and bill networks by year
## Rule to connect two together: 
### - Campaign data of Year 1 & 2, + Bill Co-Authorship Data of Year 3 & 4
### - In the future, we can try a longer span of campaign records


#### Global var
CAM_YEAR = 2000
BILL_YEAR = 2002 ## read data from processed-data/legislator_bill_edge_list.csv

pd.set_option('display.max_columns', 7)

def loadCandidateMaster():
    filePath = "../processed-data/candidate_node_mapping_manual.csv"

    result = pd.read_csv(filePath, index_col=False)

    print result.head()

    return result

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
    result = result[['CMTE_ID']]
    result = result.groupby(['CMTE_ID']).size().reset_index(name='Freq')

    result["ComNodeId"] = result.reset_index().index

    #print(result.head())

    return result

def loadContributionsYear():
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

    result = result[['CMTE_ID','TRANSACTION_DT', 'TRANSACTION_AMT', 'CAND_ID']]
    #result = result[['CMTE_ID','CAND_ID']]
    result = result.groupby(['CMTE_ID','TRANSACTION_DT', 'TRANSACTION_AMT', 'CAND_ID']).size().reset_index(name='Freq')
    ## remove the case where committees received money from candidates
    result = result[result.TRANSACTION_AMT >= 0]

    result['ContributionYear'] = (result['TRANSACTION_DT'] % 10000).apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int64)
    ##remove the case where years are invalid
    result = result[result.ContributionYear > 0]


    result["ComNodeId"] = result.groupby(['CMTE_ID']).ngroup()



    print "---Processing contributions_to_candidates data..."
    print result.head()

    return result

def findBillNodeOffset():

    offset = 0

    filePathNode = "../processed-data/"
    f = filePathNode + "bill_node.csv"

    bill = pd.read_csv(f, delimiter= ",", index_col=False)

    offset = bill['NId'].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int64).max()

    print "Committee NodeID Offset is %d" % (offset + 1)

    return offset

if __name__ == "__main__":
    
    c = loadContributionsYear()
    can = loadCandidateMaster() 

    # merge dataframes together
    c = pd.merge(c, can, on='CAND_ID', how='left')
    #c = pd.merge(c, com, on='CMTE_ID', how='left')
    print "---after merging, raw combined file looks like this: "
    print(c.head())

    #c = c[['CMTE_ID','TRANSACTION_DT', 'TRANSACTION_AMT', 'CAND_ID', 'CMTE_NM',  'CAND_NAME']]
    c = c.fillna('')
    
    c[['NodeID','ComNodeId']] = c[['NodeID','ComNodeId']].apply(pd.to_numeric, errors='coerce').fillna(-1).astype(np.int64)
    
    ## remove entries where Nid = 0, meaning unsuccessful candidates
    c = c[c.fname != '']
    c = c[c.NodeID != -1]

    #c = c[c.NodeID == -1]

    ## add differential to commmittee node IDs, based on bill network
    
    d = findBillNodeOffset()

    c['ComNodeId'] = c['ComNodeId'] + d + 1 # need to plus one so as not to collide with bills

    print "---after cleaning, raw combined file looks like this: "
    print(c.head())
    print(c.shape)
    ## save raw file
    c.to_csv("../processed-data/campaignNetworks_raw_v2.csv", index = False)

    print "---done saving  "
    ## create an undirect graph
    G = snap.TUNGraph.New()
    for index, row in c.iterrows():
        if G.IsNode(row['NodeID']) is False:
            G.AddNode(row['NodeID'])
        if G.IsNode(row['ComNodeId']) is False:
            G.AddNode(row['ComNodeId'])
        if G.IsEdge(row['NodeID'], row['ComNodeId']) is False:
            G.AddEdge(row['NodeID'], row['ComNodeId'])

    print "G node count is %d" % (G.GetNodes())
    print "G edge count is %d" % (G.GetEdges())

    snap.SaveEdgeList(G, "../processed-data/campaignNetworks_raw.csv", "Save 1981 to 2016 campaign network info as tab-separated list of edges, using unified candidate node IDs")